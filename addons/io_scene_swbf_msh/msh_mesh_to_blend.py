""" Converts msh meshes to Blender counterparts """


import bpy
import bmesh
import math

from enum import Enum
from typing import List, Set, Dict, Tuple

from .msh_scene import Scene
from .msh_material_to_blend import *
from .msh_model import *
from .msh_skeleton_utilities import *
from .msh_model_gather import get_is_model_hidden


from .crc import *

import os


def validate_segment_geometry(segment : GeometrySegment):
    if not segment.positions:
        return False
    if not segment.triangles and not segment.triangle_strips and not segment.polygons:
        return False
    if not segment.material_name:
        return False
    if not segment.normals:
        return False
    return True


def shadow_to_mesh_object(model: Model) -> bpy.types.Object:
    """ Forms a blender mesh from SHDW chunk data """

    blender_mesh = bpy.data.meshes.new(model.name)

    # Convert and collect vertices
    vertex_positions = [tuple(convert_vector_space(p)) for p in model.geometry[0].shadow.positions]

    # Reconstruct faces from the half-edge data
    faces = []
    visited_edges = set()
    edges_data = model.geometry[0].shadow.edges 

    for start_edge_idx in range(len(edges_data)):
        if start_edge_idx in visited_edges:
            continue
            
        face_vertices = []
        current_edge_idx = start_edge_idx
        
        # Follow the pointer chain around the boundary of the face
        while current_edge_idx not in visited_edges:
            visited_edges.add(current_edge_idx)
            
            curr_edge = edges_data[current_edge_idx]
            
            # curr_edge[0] is the vertex index this half-edge originates from
            face_vertices.append(curr_edge[0])
            
            # curr_edge[1] is actually the NEXT EDGE INDEX in this face loop
            next_edge_idx = curr_edge[1]
            
            # Safety check: if the file data points to an invalid index, break
            if next_edge_idx >= len(edges_data) or next_edge_idx is None:
                break
                
            current_edge_idx = next_edge_idx
            
            # If we've looped back to the starting edge, the face is complete
            if current_edge_idx == start_edge_idx:
                break
                
        # Only add valid polygons (at least 3 vertices)
        if len(face_vertices) >= 3:
            faces.append(face_vertices)

    # Populate Blender Mesh
    blender_mesh.from_pydata(vertex_positions, [], faces)
    blender_mesh.update()

    # Add new blender object with blender_mesh data
    blender_mesh_object = bpy.data.objects.new(model.name, blender_mesh)

    return blender_mesh_object


def model_to_mesh_object(model: Model, scene : Scene, materials_map : Dict[str, bpy.types.Material]) -> bpy.types.Object:

    blender_mesh = bpy.data.meshes.new(model.name)

    # Per vertex data which will eventually be remapped to loops
    vertex_positions = []
    vertex_uvs = []
    vertex_normals = []
    vertex_colors = []

    # Keeps track of which vertices each group of weights affects
    # i.e. maps offset of vertices -> weights that affect them
    vertex_weights_offsets = {}

    # Since polygons in a msh segment index into the segment's verts,
    # we must keep an offset to index them into the verts of the whole mesh
    polygon_index_offset = 0

    # List of tuples of face indices
    polygons = []

    # Each polygon has an index into the mesh's material list
    current_material_index = 0
    polygon_material_indices = []


    if model.geometry:
        geometry_has_colors = any(segment.colors for segment in model.geometry)

        for segment in model.geometry:

            if not validate_segment_geometry(segment):
                continue

            blender_mesh.materials.append(materials_map[segment.material_name])

            vertex_positions += [tuple(convert_vector_space(p)) for p in segment.positions]

            if segment.texcoords:
                vertex_uvs += [tuple(texcoord) for texcoord in segment.texcoords]
            else:
                vertex_uvs += [(0.0,0.0) for _ in range(len(segment.positions))]

            if segment.normals:
                vertex_normals += [tuple(convert_vector_space(n)) for n in segment.normals]

            if segment.colors:
                vertex_colors.extend(segment.colors)
            elif geometry_has_colors:
                [vertex_colors.extend([0.0, 0.0, 0.0, 1.0]) for _ in range(len(segment.positions))]
            
            if segment.weights:
                vertex_weights_offsets[polygon_index_offset] = segment.weights


            segment_polygons = []

            if segment.triangles:
                segment_polygons = [tuple([ind + polygon_index_offset for ind in tri]) for tri in segment.triangles]
            elif segment.triangle_strips:
                winding = [0,1,2]
                rwinding = [1,0,2]
                for strip in segment.triangle_strips:
                    for i in range(len(strip) - 2):
                        strip_tri = tuple([polygon_index_offset + strip[i+j] for j in (winding if i % 2 == 0 else rwinding)])
                        segment_polygons.append(strip_tri)
            elif segment.polygons:
                segment_polygons = [tuple([ind + polygon_index_offset for ind in polygon]) for polygon in segment.polygons]

            polygon_index_offset += len(segment.positions)

            polygons += segment_polygons

            polygon_material_indices += [current_material_index for _ in segment_polygons]
            current_material_index += 1

        '''
        Start building the blender mesh
        '''

        # VERTICES

        # This is all we have to do for vertices, other attributes are done per-loop
        blender_mesh.vertices.add(len(vertex_positions))
        blender_mesh.vertices.foreach_set("co", [component for vertex_position in vertex_positions for component in vertex_position])

        # LOOPS 

        flat_indices = [index for polygon in polygons for index in polygon]

        blender_mesh.loops.add(len(flat_indices))

        # Position indices
        blender_mesh.loops.foreach_set("vertex_index", flat_indices)

        # Normals
        blender_mesh.loops.foreach_set("normal", [component for i in flat_indices for component in vertex_normals[i]])

        # UVs
        blender_mesh.uv_layers.new(do_init=False)
        blender_mesh.uv_layers[0].data.foreach_set("uv", [component for i in flat_indices for component in vertex_uvs[i]])

        # Colors
        if geometry_has_colors:
            blender_mesh.color_attributes.new("COLOR0", "FLOAT_COLOR", "POINT")
            blender_mesh.color_attributes[0].data.foreach_set("color", vertex_colors)


        # POLYGONS/FACES

        blender_mesh.polygons.add(len(polygons))

        # Indices of starting loop for each polygon
        polygon_loop_start_indices = []
        current_polygon_start_index = 0

        # Number of loops in this polygon.  Polygon i will use
        # loops from polygon_loop_start_indices[i] to 
        # polygon_loop_start_indices[i] + polygon_loop_totals[i]
        polygon_loop_totals = []

        for polygon in polygons:
            polygon_loop_start_indices.append(current_polygon_start_index)

            current_polygon_length = len(polygon)
            current_polygon_start_index += current_polygon_length

            polygon_loop_totals.append(current_polygon_length)

        blender_mesh.polygons.foreach_set("loop_start", polygon_loop_start_indices)
        blender_mesh.polygons.foreach_set("loop_total", polygon_loop_totals)
        blender_mesh.polygons.foreach_set("material_index", polygon_material_indices)
        blender_mesh.polygons.foreach_set("use_smooth", [True for _ in polygons])

        blender_mesh.validate(clean_customdata=False) 
        blender_mesh.update()


        # Reset custom normals after calling update/validate
        reset_normals = [0.0] * (len(blender_mesh.loops) * 3)
        blender_mesh.loops.foreach_get("normal", reset_normals)
        blender_mesh.normals_split_custom_set(tuple(zip(*(iter(reset_normals),) * 3)))


    blender_mesh_object = bpy.data.objects.new(model.name, blender_mesh)


    # VERTEX GROUPS

    vertex_groups_indicies = {}

    for offset in vertex_weights_offsets:
        for i, weight_set in enumerate(vertex_weights_offsets[offset]):
            for weight in weight_set:
                index = weight.bone

                if index not in vertex_groups_indicies:
                    model_name = scene.models[index].name
                    vertex_groups_indicies[index] = blender_mesh_object.vertex_groups.new(name=model_name)

                vertex_groups_indicies[index].add([offset + i], weight.weight, 'ADD')

    # Cleanup mesh (Duplicate vertices, normals)
    if model.geometry:
        # 1. Store original custom loop normals mapped by a rounded vertex coordinate
        custom_normal_map = {}
        
        for poly in blender_mesh.polygons:
            for loop_idx in poly.loop_indices:
                loop = blender_mesh.loops[loop_idx]
                vert = blender_mesh.vertices[loop.vertex_index]
                v_key = tuple(round(c, 4) for c in vert.co)
                
                # Fetch normal data
                corner_normal = blender_mesh.corner_normals[loop_idx].vector
                
                if v_key not in custom_normal_map:
                    custom_normal_map[v_key] = []
                custom_normal_map[v_key].append((corner_normal.copy(), poly.normal.copy()))

        # 2. Perform the BMesh background cleanup safely
        bm = bmesh.new()
        bm.from_mesh(blender_mesh)
        
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)

        for layer in list(bm.loops.layers.float_vector.values()):
            bm.loops.layers.float_vector.remove(layer)

        bm.to_mesh(blender_mesh)
        bm.free()

        # 3. Project preserved custom normals back onto the newly welded topology
        new_loop_normals = []
        
        for poly in blender_mesh.polygons:
            for loop_idx in poly.loop_indices:
                loop = blender_mesh.loops[loop_idx]
                vert = blender_mesh.vertices[loop.vertex_index]
                v_key = tuple(round(c, 4) for c in vert.co)
                
                # Use the newly cached corner normal as a base fallback
                matched_normal = blender_mesh.corner_normals[loop_idx].vector
                
                if v_key in custom_normal_map:
                    # Find the closest matching face normal to preserve the sharp edge boundary
                    best_dot = -1.0
                    for orig_loop_norm, orig_poly_norm in custom_normal_map[v_key]:
                        dot = poly.normal.dot(orig_poly_norm)
                        if dot > best_dot:
                            best_dot = dot
                            matched_normal = orig_loop_norm
                
                new_loop_normals.append(matched_normal)
                
        # 4. Set the reconstructed custom split normals back into the mesh
        if new_loop_normals:
            blender_mesh.normals_split_custom_set(new_loop_normals)

    return blender_mesh_object
