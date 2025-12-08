""" Gathers the Blender objects from the current scene and returns them as a list of
    Model objects. """

import ast
import os
import bpy
import math
from enum import Enum
from typing import List, Set, Dict, Tuple
from itertools import zip_longest
from .msh_model import *
from .msh_model_utilities import *
from .msh_utilities import *
from .msh_skeleton_utilities import *

SKIPPED_OBJECT_TYPES = {"LATTICE", "CAMERA", "LIGHT", "SPEAKER", "LIGHT_PROBE"}
MESH_OBJECT_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT", "GPENCIL"}
MAX_MSH_VERTEX_COUNT = 32767

def gather_models(apply_modifiers: bool, export_target: str, skeleton_only: bool) -> Tuple[List[Model], bpy.types.Object]:
    """ Gathers the Blender objects from the current scene and returns them as a list of
        Model objects. """

    depsgraph = bpy.context.evaluated_depsgraph_get()
    parents = create_parents_set()

    models_list: List[Model] = []

    # Composite bones are bones which have geometry.  
    # If a child object has the same name, it will take said child's geometry.

    # Pure bones are just bones and after all objects are explored the only
    # entries remaining in this dict will be bones without geometry.  
    pure_bones_from_armature = {}
    armature_found = None

    # Non-bone objects that will be exported
    blender_objects_to_export = []

    # This must be seperate from the list above,
    # since exported objects will contain Blender objects as well as bones
    # Here we just keep track of all names, regardless of origin
    exported_object_names: Set[str] = set() 

    # Me must keep track of hidden objects separately because
    # evaluated_get clears hidden status
    blender_objects_to_hide: Set[str] = set()

    # Armature must be processed before everything else!

    # In this loop we also build a set of names of all objects
    # that will be exported.  This is necessary so we can prune vertex
    # groups that do not reference exported objects in the main 
    # model building loop below this one.
    for uneval_obj in select_objects(export_target):

        if get_is_model_hidden(uneval_obj):
            blender_objects_to_hide.add(uneval_obj.name)

        if uneval_obj.type == "ARMATURE" and not armature_found:
            # Keep track of the armature, we don't want to process > 1!
            armature_found = uneval_obj.evaluated_get(depsgraph) if apply_modifiers else uneval_obj
            # Get all bones in a separate list.  While we iterate through
            # objects we removed bones with geometry from this dict.  After iteration
            # is done, we add the remaining bones to the models from exported
            # scene objects.
            pure_bones_from_armature = expand_armature(armature_found)
            # All bones to set
            exported_object_names.update(pure_bones_from_armature.keys())
        
        elif not (uneval_obj.type in SKIPPED_OBJECT_TYPES and uneval_obj.name not in parents):
            exported_object_names.add(uneval_obj.name)
            blender_objects_to_export.append(uneval_obj)
        
        else:
            pass

    for uneval_obj in blender_objects_to_export:

        obj = uneval_obj.evaluated_get(depsgraph) if apply_modifiers else uneval_obj

        check_for_bad_lod_suffix(obj)

        # Test for a mesh object that should be a BONE on export.
        # If so, we inject geometry into the BONE while not modifying it's transform/name
        # and remove it from the set of BONES without geometry (pure).
        if obj.name in pure_bones_from_armature:
            model = pure_bones_from_armature.pop(obj.name)
        else:
            model = Model()
            model.name = obj.name
            model.model_type = ModelType.NULL if skeleton_only else get_model_type(obj, armature_found)

            transform = obj.matrix_local

            if obj.parent_bone:
                model.parent = obj.parent_bone

                # matrix_local, when called on an armature child also parented to a bone, appears to be broken.
                # At the very least, the results contradict the docs...  
                armature_relative_transform = obj.parent.matrix_world.inverted() @ obj.matrix_world
                transform = obj.parent.data.bones[obj.parent_bone].matrix_local.inverted() @ armature_relative_transform 

            else:
                if obj.parent is not None:
                    if obj.parent.type == "ARMATURE":
                        model.parent = obj.parent.parent.name if obj.parent.parent else ""
                        transform = obj.parent.matrix_local @ transform
                    else:
                        model.parent = obj.parent.name

            local_translation, local_rotation, _ = transform.decompose()
            model.transform.rotation = convert_rotation_space(local_rotation)  
            model.transform.translation = convert_vector_space(local_translation)

        if obj.type in MESH_OBJECT_TYPES and not skeleton_only:

            if model.model_type == ModelType.CLOTH:
                model.cloth = cloth_from_object(obj)
                # Cloth models do not have standard geometry segments
                model.geometry = None

            # Vertex groups are often used for purposes other than skinning.
            # Here we gather all vgroups and select the ones that reference
            # objects included in the export.
            valid_vgroup_indices : Set[int] = set()
            if model.model_type == ModelType.SKIN:
                valid_vgroups = [group for group in obj.vertex_groups if group.name in exported_object_names]
                valid_vgroup_indices = { group.index for group in valid_vgroups }
                model.bone_map = [ group.name for group in valid_vgroups ]

            if model.model_type != ModelType.CLOTH:
                mesh = obj.to_mesh()
                model.geometry = create_mesh_geometry(mesh, valid_vgroup_indices)

            obj.to_mesh_clear()

            _, _, world_scale = obj.matrix_world.decompose()
            world_scale = convert_scale_space(world_scale)
            if model.geometry:
                scale_segments(world_scale, model.geometry)

            if model.geometry:
                for segment in model.geometry:
                    if len(segment.positions) > MAX_MSH_VERTEX_COUNT:
                        raise RuntimeError(f"Object '{obj.name}' has resulted in a .msh geometry segment that has "
                                           f"more than {MAX_MSH_VERTEX_COUNT} vertices! Split the object's mesh up "
                                           f"and try again!")

        if get_is_collision_primitive(obj):
            model.collisionprimitive = get_collision_primitive(obj)

        model.hidden = model.name in blender_objects_to_hide

        models_list.append(model)

    # We removed all composite bones after looking through the objects,
    # so the bones left are all pure and we add them all here.
    return (models_list + list(pure_bones_from_armature.values()), armature_found)


def cloth_from_object(blender_obj: bpy.types.Object) -> Cloth:
    """ Gathers cloth data from a Blender mesh object. """
    if blender_obj.type != 'MESH':
        return None

    cloth = Cloth()
    mesh = blender_obj.data

    # Get texture from the first material slot
    if mesh.materials and mesh.materials[0]:
        material = mesh.materials[0]
        texture_found = False

        # Try to get texture from a standard node setup first
        if material.use_nodes and material.node_tree:
            for node in material.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    base_color_input = node.inputs.get("Base Color")
                    if base_color_input and base_color_input.is_linked:
                        linked_node = base_color_input.links[0].from_node
                        if linked_node.type == 'TEX_IMAGE' and linked_node.image:
                            cloth.texture = os.path.basename(linked_node.image.filepath)
                            texture_found = True
                            break # Found it

        # Fallback to custom property if no standard texture was found
        if not texture_found and hasattr(material, 'swbf_msh_mat'):
            mat_props = material.swbf_msh_mat
            if mat_props.diffuse_map:
                cloth.texture = os.path.basename(mat_props.diffuse_map)
    else:
        raise RuntimeError(f"Object '{blender_obj.name}' has no materials!")
    
    # Get vertex positions (converted to SWBF coordinate space)
    cloth.positions = [convert_vector_space(v.co) for v in mesh.vertices]

    # Get UVs from the active UV layer
    if mesh.uv_layers.active:
        uv_layer = mesh.uv_layers.active.data
        # Initialize UV list with correct size
        cloth.uvs = [[0.0, 0.0]] * len(mesh.vertices)
        # Use loops to find the UV for each vertex
        for poly in mesh.polygons:
            for loop_index in poly.loop_indices:
                loop = mesh.loops[loop_index]
                vert_index = loop.vertex_index
                cloth.uvs[vert_index] = list(uv_layer[loop_index].uv)
    else:
        raise RuntimeError(f"Object '{blender_obj.name}' has no UVs!")

    # Get triangles
    mesh.calc_loop_triangles()
    cloth.triangles = [[tri.vertices[0], tri.vertices[1], tri.vertices[2]] for tri in mesh.loop_triangles]

    # Get pinned vertices and their bone weights from vertex groups
    pin_group = blender_obj.vertex_groups.get("Pin")
    if pin_group:
        pinned_verts = {}  # Dict of {vertex_index: bone_name}
        for v in mesh.vertices:
            is_pinned = any(g.group == pin_group.index and g.weight > 0.5 for g in v.groups)
            if is_pinned:
                bone_name = None
                highest_weight = 0.0
                for g2 in v.groups:
                    group_name = blender_obj.vertex_groups[g2.group].name
                    if group_name != "Pin" and g2.weight > highest_weight:
                        highest_weight = g2.weight
                        bone_name = group_name
                if bone_name:
                    pinned_verts[v.index] = bone_name

        if pinned_verts:
            sorted_pins = sorted(pinned_verts.items())
            cloth.fixed_points = [item[0] for item in sorted_pins]
            cloth.fixed_weights_bones = [item[1] for item in sorted_pins]

    # Get collision primitives for this cloth
    raw = blender_obj.get("swbf_msh_cloth_collisions", "[]")

    try:
        names = ast.literal_eval(raw)  # ['Cube', 'Sphere', 'Cylinder']
    except (ValueError, SyntaxError):
        names = []

    # Clear internal list of collision_objects (Might have changed)
    cloth.collision_objects = []

    for name in names:
        if name in bpy.context.scene.objects:
            if get_is_cloth_collision_primitive(bpy.context.scene.objects[name]):
                obj = bpy.context.scene.objects[name]
                prim = get_cloth_collision_primitive(obj)
                prim.name = obj.name

                if obj.parent.name == "skeleton":
                    prim.parent_name = obj.parent_bone
                else:
                    prim.parent_name = obj.parent.name
                
                cloth.collision_objects.append(prim)

    # If empty swbf_msh_cloth_collisions property, search entire scene
    if not names:
        for obj in bpy.context.scene.objects:
            if get_is_cloth_collision_primitive(obj):
                prim = get_cloth_collision_primitive(obj)
                # The primitive needs its name and parent name for the COLL chunk
                prim.name = obj.name

                # Scenes with skeletons need to target parent_bone instead of parent.name
                if obj.parent.name == "skeleton":
                    prim.parent_name = obj.parent_bone
                else:
                    prim.parent_name = obj.parent.name
                
                cloth.collision_objects.append(prim)

    # Check if the mesh has been edited by comparing topology signatures.
    current_signature = f"{len(mesh.vertices)}:{len(mesh.edges)}:{len(mesh.polygons)}"
    original_signature = blender_obj.get("swbf_msh_cloth_mesh_signature", "")
    mesh_is_unedited = (current_signature == original_signature)

    # If mesh is unedited, just use imported data if present.
    if mesh_is_unedited:
        if "swbf_msh_cloth_stretch_constraints" in blender_obj and "swbf_msh_cloth_cross_constraints" in blender_obj and "swbf_msh_cloth_bend_constraints" in blender_obj:
            try:
                cloth.stretch_constraints = ast.literal_eval(blender_obj["swbf_msh_cloth_stretch_constraints"])
                cloth.cross_constraints = ast.literal_eval(blender_obj["swbf_msh_cloth_cross_constraints"])
                cloth.bend_constraints = ast.literal_eval(blender_obj["swbf_msh_cloth_bend_constraints"])
                return cloth
            except (ValueError, SyntaxError):
                # If parsing fails, proceed to recalculate values
                pass

    # Recalculated constraints
    stretch_constraints = set()
    cross_constraints = set()
    bend_constraints = set()

    fixed_point_set = set(cloth.fixed_points)

    def is_fixed(idx):
        return idx in fixed_point_set

    polygons = mesh.polygons

    # Calculate cloth constraints
    for i, face in enumerate(polygons):
        # v1 ┌────┐ v2
        #    │    │
        #    │    │
        # v4 └────┘ v3
        # Stretch constraints, along the boundary of the polygon
        # v1-v2, v2-v3, v3-v4, v4-v1
        last_vert_idx = face.vertices[-1]
        for vert_idx in face.vertices:
            if is_fixed(vert_idx) and is_fixed(last_vert_idx):
                last_vert_idx = vert_idx
                continue

            constraint = tuple(sorted((last_vert_idx, vert_idx)))
            stretch_constraints.add(constraint)
            last_vert_idx = vert_idx

        # Cross constraints, diagonally across quad
        # v1-v3, v2-v4
        # Only works for quads
        if len(face.vertices) == 4:
            v = face.vertices
            pair_1 = (v[0], v[2])
            pair_2 = (v[1], v[3])

            # Only add constraint if either or both vertices are dynamic
            if not (is_fixed(pair_1[0]) and is_fixed(pair_1[1])):
                cross_constraints.add(tuple(sorted(pair_1)))

            if not (is_fixed(pair_2[0]) and is_fixed(pair_2[1])):
                cross_constraints.add(tuple(sorted(pair_2)))

        # Bend constraints
        # Prevents cloth from overbending
        # v1   v2     v3
        # ┌─────┬─────┐
        # │     │     │
        # │ v8  │ v9  │ v4
        # ├─────┼─────┤
        # │     │     │
        # │     │     │
        # └─────┴─────┘
        # v7    v6    v5
        #
        # v1-v3, v3-v5, v5-v7, v7-v1, v8-v4, v2-v6
        # Connects all faces in this pattern
        for face2 in polygons:
            if face is face2:
                continue

            shared_vertices = [v for v in face2.vertices if v in face.vertices]
            if len(shared_vertices) != 2:
                continue

            non_shared1 = [v for v in face.vertices if v not in shared_vertices]
            non_shared2 = [v for v in face2.vertices if v not in shared_vertices]
            if not non_shared1 or not non_shared2:
                continue

            if len(non_shared1) == 1 and len(non_shared2) == 1:
                # Triangle case
                v1, v2 = non_shared1[0], non_shared2[0]
                if not (is_fixed(v1) and is_fixed(v2)):
                    bend_constraints.add(tuple(sorted((v1, v2))))
            else:
                # Quad case: pair by adjacency
                pairs = []
                for v1 in non_shared1:
                    for v2 in non_shared2:
                        # Check if v1 and v2 are opposite across the shared edge
                        if any(frozenset((v1, sv)) in face.edge_keys and
                            frozenset((v2, sv)) in face2.edge_keys
                            for sv in shared_vertices):
                            pairs.append((v1, v2))

                for v1, v2 in pairs:
                    if not (is_fixed(v1) and is_fixed(v2)):
                        bend_constraints.add(tuple(sorted((v1, v2))))

    # Save recalculated constraints
    cloth.stretch_constraints = [list(item) for item in stretch_constraints]
    cloth.cross_constraints = [list(item) for item in cross_constraints]
    cloth.bend_constraints = [list(item) for item in bend_constraints]

    return cloth


def create_parents_set() -> Set[str]:
    """ Creates a set with the names of the Blender objects from the current scene
        that have at least one child. """
        
    parents = set()

    for obj in bpy.context.scene.objects:
        if obj.parent is not None:
            parents.add(obj.parent.name)

    return parents


def create_mesh_geometry(mesh: bpy.types.Mesh, valid_vgroup_indices: Set[int]) -> List[GeometrySegment]:
    """ Creates a list of GeometrySegment objects from a Blender mesh.
        Does NOT create triangle strips in the GeometrySegment however. """

    mesh.validate_material_indices()
    mesh.calc_loop_triangles()

    material_count = max(len(mesh.materials), 1)

    segments: List[GeometrySegment] = [GeometrySegment() for i in range(material_count)]
    vertex_cache = [dict() for i in range(material_count)]
    vertex_remap: List[Dict[Tuple[int, int], int]] = [dict() for i in range(material_count)]
    polygons: List[Set[int]] = [set() for i in range(material_count)]

    if mesh.color_attributes.active_color is not None:
        for segment in segments:
            segment.colors = []

    if valid_vgroup_indices:
        for segment in segments:
            segment.weights = []

    for segment, material in zip(segments, mesh.materials):
        segment.material_name = material.name


    def add_vertex(material_index: int, vertex_index: int, loop_index: int) -> int:
        nonlocal segments, vertex_remap

        vertex_cache_miss_index = -1
        segment = segments[material_index]
        cache = vertex_cache[material_index]
        remap = vertex_remap[material_index]

        # always use loop normals since we always calculate a custom split set        
        vertex_normal = Vector( mesh.loops[loop_index].normal )


        def get_cache_vertex():
            yield mesh.vertices[vertex_index].co.x
            yield mesh.vertices[vertex_index].co.y
            yield mesh.vertices[vertex_index].co.z

            yield vertex_normal.x
            yield vertex_normal.y
            yield vertex_normal.z

            if mesh.uv_layers.active is not None:
                yield mesh.uv_layers.active.data[loop_index].uv.x
                yield mesh.uv_layers.active.data[loop_index].uv.y

            if segment.colors is not None:
                active_color = mesh.color_attributes.active_color
                data_index = loop_index if active_color.domain == "CORNER" else vertex_index

                for v in mesh.color_attributes.active_color.data[data_index].color:
                    yield v

            if segment.weights is not None:
                for v in mesh.vertices[vertex_index].groups:
                    if v.group in valid_vgroup_indices:                    
                        yield v.group
                        yield v.weight

        vertex_cache_entry = tuple(get_cache_vertex())
        cached_vertex_index = cache.get(vertex_cache_entry, vertex_cache_miss_index)

        if cached_vertex_index != vertex_cache_miss_index:
            remap[(vertex_index, loop_index)] = cached_vertex_index

            return cached_vertex_index

        new_index: int = len(segment.positions)
        cache[vertex_cache_entry] = new_index
        remap[(vertex_index, loop_index)] = new_index

        segment.positions.append(convert_vector_space(mesh.vertices[vertex_index].co))
        segment.normals.append(convert_vector_space(vertex_normal))

        if mesh.uv_layers.active is None:
            segment.texcoords.append(Vector((0.0, 0.0)))
        else:
            segment.texcoords.append(mesh.uv_layers.active.data[loop_index].uv.copy())

        if segment.colors is not None:
            active_color = mesh.color_attributes.active_color
            data_index = loop_index if active_color.domain == "CORNER" else vertex_index

            segment.colors.append(list(active_color.data[data_index].color))

        if segment.weights is not None:
            groups = mesh.vertices[vertex_index].groups
            segment.weights.append([VertexWeight(v.weight, v.group) for v in groups if v.group in valid_vgroup_indices])

        return new_index

    for tri in mesh.loop_triangles:
        polygons[tri.material_index].add(tri.polygon_index)
        segments[tri.material_index].triangles.append([
            add_vertex(tri.material_index, tri.vertices[0], tri.loops[0]),
            add_vertex(tri.material_index, tri.vertices[1], tri.loops[1]),
            add_vertex(tri.material_index, tri.vertices[2], tri.loops[2])])

    for segment, remap, polys in zip(segments, vertex_remap, polygons):
        for poly_index in polys:
            poly = mesh.polygons[poly_index]

            segment.polygons.append([remap[(v, l)] for v, l in zip(poly.vertices, poly.loop_indices)])

    return segments


def get_model_type(obj: bpy.types.Object, armature_found: bpy.types.Object) -> ModelType:
    """ Get the ModelType for a Blender object. """

    # A cloth object is identified by its "Pin" vertex group
    if "Pin" in obj.vertex_groups.keys():
        return ModelType.CLOTH

    if obj.type in MESH_OBJECT_TYPES:
        # Objects can have vgroups for non-skinning purposes.
        # If we can find one vgroup that shares a name with a bone in the 
        # armature, we know the vgroup is for weighting purposes and thus
        # the object is a skin.  Otherwise, interpret it as a static mesh.

        # We must also check that an armature included in the export
        # and that it is the same one this potential skin is weighting to.
        # If we failed to do this, a user could export a selected object
        # that is a skin, but the weight data in the export would reference
        # nonexistent models!
        if (obj.vertex_groups and armature_found and 
            obj.parent and obj.parent.name == armature_found.name):
            
            for vgroup in obj.vertex_groups:
                if vgroup.name in armature_found.data.bones:
                    return ModelType.SKIN

            return ModelType.STATIC
        
        else:
            return ModelType.STATIC

    return ModelType.NULL


def get_is_model_hidden(obj: bpy.types.Object) -> bool:
    """ Gets if a Blender object should be marked as hidden in the .msh file. """

    if obj.hide_get():
        return True

    name = obj.name.lower()

    if name.startswith("c_"):
        return True
    if name.startswith("sv_"):
        return True
    if name.startswith("p_"):
        return True
    if name.startswith("collision"):
        return True

    if obj.type not in MESH_OBJECT_TYPES:
        return True

    if name.endswith("_lod2"):
        return True
    if name.endswith("_lod3"):
        return True
    if name.endswith("_lowrez"):
        return True
    if name.endswith("_lowres"):
        return True

    return False


def get_is_collision_primitive(obj: bpy.types.Object) -> bool:
    """ Gets if a Blender object represents a collision primitive. """

    name = obj.name.lower()

    return name.startswith("p_")


def get_is_cloth_collision_primitive(obj: bpy.types.Object) -> bool:
    """ Gets if a Blender object represents a collision primitive. """

    name = obj.name.lower()

    return name.startswith("c_")


def get_collision_primitive(obj: bpy.types.Object) -> CollisionPrimitive:
    """ Gets the CollisionPrimitive of an object or raises an error if
        it can't. """

    primitive = CollisionPrimitive()
    primitive.shape = get_collision_primitive_shape(obj)

    if primitive.shape == CollisionPrimitiveShape.SPHERE:
        # Tolerate a 5% difference to account for icospheres with 2 subdivisions.
        if not (math.isclose(obj.dimensions[0], obj.dimensions[1], rel_tol=0.05) and
                math.isclose(obj.dimensions[0], obj.dimensions[2], rel_tol=0.05)):
            raise RuntimeError(f"Object '{obj.name}' is being used as a sphere collision "
                               f"primitive but it's dimensions are not uniform!")

        primitive.radius = max(obj.dimensions[0], obj.dimensions[1], obj.dimensions[2]) * 0.5
    elif primitive.shape == CollisionPrimitiveShape.CYLINDER:
        primitive.radius = max(obj.dimensions[0], obj.dimensions[1]) * 0.5
        primitive.height = obj.dimensions[2]
    elif primitive.shape == CollisionPrimitiveShape.BOX:
        primitive.radius = obj.dimensions[0] * 0.5
        primitive.height = obj.dimensions[2] * 0.5
        primitive.length = obj.dimensions[1] * 0.5

    return primitive


def get_cloth_collision_primitive(obj: bpy.types.Object) -> ClothCollisionPrimitive:
    """ Gets the ClothCollisionPrimitive of an object or raises an error if
        it can't. """

    primitive = ClothCollisionPrimitive()
    primitive.shape = get_cloth_collision_primitive_shape(obj)

    if primitive.shape == ClothCollisionPrimitiveShape.SPHERE:
        # Tolerate a 5% difference to account for icospheres with 2 subdivisions.
        if not (math.isclose(obj.dimensions[0], obj.dimensions[1], rel_tol=0.05) and
                math.isclose(obj.dimensions[0], obj.dimensions[2], rel_tol=0.05)):
            raise RuntimeError(f"Object '{obj.name}' is being used as a sphere collision "
                               f"primitive but it's dimensions are not uniform!")

        primitive.radius = max(obj.dimensions[0], obj.dimensions[1], obj.dimensions[2]) * 0.5
    elif primitive.shape == ClothCollisionPrimitiveShape.CYLINDER:
        primitive.radius = max(obj.dimensions[0], obj.dimensions[1]) * 0.5
        primitive.height = obj.dimensions[2]
    elif primitive.shape == ClothCollisionPrimitiveShape.BOX:
        primitive.radius = obj.dimensions[0] * 0.5
        primitive.height = obj.dimensions[2] * 0.5
        primitive.length = obj.dimensions[1] * 0.5

    return primitive


def get_collision_primitive_shape(obj: bpy.types.Object) -> CollisionPrimitiveShape:
    """ Gets the CollisionPrimitiveShape of an object or raises an error if
        it can't. """

    # arc170 fighter has examples of box colliders without proper naming
    # and cis_hover_aat has a cylinder which is named p_vehiclesphere.
    # To export these properly we must check the collision_prim property
    # that was assigned on import BEFORE looking at the name.
    prim_type = obj.swbf_msh_coll_prim.prim_type
    if prim_type in [item.value for item in CollisionPrimitiveShape]:
        return CollisionPrimitiveShape(prim_type)

    name = obj.name.lower()

    if "sphere" in name or "sphr" in name or "spr" in name:
        return CollisionPrimitiveShape.SPHERE
    if "cylinder" in name or "cyln" in name or "cyl" in name:
        return CollisionPrimitiveShape.CYLINDER
    if "box" in name or "cube" in name or "cuboid" in name:
        return CollisionPrimitiveShape.BOX

    raise RuntimeError(f"Object '{obj.name}' has no primitive type specified in it's name!")


def get_cloth_collision_primitive_shape(obj: bpy.types.Object) -> ClothCollisionPrimitiveShape:
    """ Gets the ClothCollisionPrimitiveShape of an object or raises an error if
        it can't. """

    name = obj.name.lower()

    # The easy but unlikely way
    if "sphere" in name or "sphr" in name or "spr" in name:
        return ClothCollisionPrimitiveShape.SPHERE
    if "cylinder" in name or "cyln" in name or "cyl" in name:
        return ClothCollisionPrimitiveShape.CYLINDER
    if "box" in name or "cube" in name or "cuboid" in name:
        return ClothCollisionPrimitiveShape.BOX

    # Make a guess as to shape from geometry
    mesh = obj.data
    vcount = len(mesh.vertices)
    fcount = len(mesh.polygons)
    
    # Blender uvspheres
    if vcount == 482 and fcount == 512:
        return ClothCollisionPrimitiveShape.SPHERE

    #Blender icospheres
    if vcount == 42 and fcount == 80:
        return ClothCollisionPrimitiveShape.SPHERE
    
    # Blender cylinders
    if vcount == 64 and fcount == 124:
        return ClothCollisionPrimitiveShape.CYLINDER

    # XSI Spheres (original and imported/triangulated plus odd varieties found in stock models)
    if vcount == 58 and fcount == 64 or vcount == 58 and fcount == 112 or vcount == 282 and fcount == 760 or vcount == 554 and fcount == 1104:
        return ClothCollisionPrimitiveShape.SPHERE
    
    # XSI cylinders (original and imported/triangulated)
    if vcount == 42 and fcount == 48 or vcount == 58 and fcount == 80 or vcount == 14 and fcount == 24:
        return ClothCollisionPrimitiveShape.CYLINDER
    
    # XSI and Blender boxes (original and imported/triangulated)
    if vcount == 8 and fcount == 6 or vcount == 8 and fcount == 12:
        return ClothCollisionPrimitiveShape.BOX
    
    # Last resort, heuristic calculation
    # Sphere
    DIST_STD_RATIO_THRESH = 0.03

    # gather verts in object space
    verts = [v.co for v in mesh.vertices]
    V = len(verts)
    if V == 0:
        raise RuntimeError(f"Object '{obj.name}' has no geometry!")

    # centroid
    cx = sum(v.x for v in verts) / V
    cy = sum(v.y for v in verts) / V
    cz = sum(v.z for v in verts) / V

    # distances and population stddev
    dists = []
    for v in verts:
        dx = v.x - cx; dy = v.y - cy; dz = v.z - cz
        dists.append(math.sqrt(dx*dx + dy*dy + dz*dz))
    mean_d = sum(dists) / V
    var_d = sum((d - mean_d) ** 2 for d in dists) / V
    std_d = math.sqrt(var_d)
    dist_std_ratio = std_d / mean_d if mean_d else float('inf')
    is_sphere = dist_std_ratio <= DIST_STD_RATIO_THRESH

    if is_sphere:
        return ClothCollisionPrimitiveShape.SPHERE
    
    # Cube
    DIM_EQUAL_TOL = 0.03    # relative tolerance for dx,dy,dz equality
    PLANE_TOL_FRAC = 0.02   # tolerance as fraction of max half-extent
    MIN_FACE_VERTEX_RATIO = 0.95

    # gather verts
    verts = [v.co for v in mesh.vertices]
    V = len(verts)
    if V == 0:
        raise RuntimeError(f"Object '{obj.name}' has no geometry!")

    # axis-aligned bbox and center (object space)
    xs = [v.x for v in verts]; ys = [v.y for v in verts]; zs = [v.z for v in verts]
    minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys); minz, maxz = min(zs), max(zs)
    dx, dy, dz = maxx - minx, maxy - miny, maxz - minz
    mean_dim = (dx + dy + dz) / 3.0
    dims_equal = (abs(dx - mean_dim) / mean_dim <= DIM_EQUAL_TOL and
                abs(dy - mean_dim) / mean_dim <= DIM_EQUAL_TOL and
                abs(dz - mean_dim) / mean_dim <= DIM_EQUAL_TOL)

    cx, cy, cz = (minx + maxx) / 2.0, (miny + maxy) / 2.0, (minz + maxz) / 2.0
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    plane_tol = max(hx, hy, hz) * PLANE_TOL_FRAC

    # count vertices that lie on one of the six face planes (within tolerance)
    face_count = 0
    for v in verts:
        on_x_face = abs(abs(v.x - cx) - hx) <= plane_tol
        on_y_face = abs(abs(v.y - cy) - hy) <= plane_tol
        on_z_face = abs(abs(v.z - cz) - hz) <= plane_tol
        if on_x_face or on_y_face or on_z_face:
            face_count += 1

    face_ratio = face_count / V
    is_box = dims_equal and (face_ratio >= MIN_FACE_VERTEX_RATIO)

    if is_box:
        return ClothCollisionPrimitiveShape.BOX
    
    # Cylinder
    RAD_STD_RATIO_THRESH = 0.05   # radial stddev / mean
    CAP_VERTEX_RATIO = 0.20       # fraction of verts on caps (each cap)
    CAP_TOL_FRAC = 0.02           # tolerance relative to half-height for cap detection
    MIN_HEIGHT_TO_RADIUS = 0.6    # height should be at least this multiple of radius (avoid flat discs)

    # gather verts in object space
    verts = [v.co for v in mesh.vertices]
    V = len(verts)
    if V == 0:
        raise RuntimeError(f"Object '{obj.name}' has no geometry!")

    # axis-aligned bbox to pick cylinder axis (largest extent)
    xs = [v.x for v in verts]; ys = [v.y for v in verts]; zs = [v.z for v in verts]
    minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys); minz, maxz = min(zs), max(zs)
    dx, dy, dz = maxx - minx, maxy - miny, maxz - minz
    # choose axis index: 0=x,1=y,2=z
    if dx >= dy and dx >= dz:
        axis = 0; min_a, max_a = minx, maxx
    elif dy >= dx and dy >= dz:
        axis = 1; min_a, max_a = miny, maxy
    else:
        axis = 2; min_a, max_a = minz, maxz

    # center and half-height
    center_a = (min_a + max_a) / 2.0
    half_h = (max_a - min_a) / 2.0
    cap_tol = max(half_h, 1e-6) * CAP_TOL_FRAC

    # compute radial distances from chosen axis and cap membership
    rads = []
    cap_count = 0
    for v in verts:
        if axis == 0:
            a = v.x - center_a
            ox, oy = v.y, v.z
        elif axis == 1:
            a = v.y - center_a
            ox, oy = v.x, v.z
        else:
            a = v.z - center_a
            ox, oy = v.x, v.y
        r = math.hypot(ox, oy)
        rads.append(r)
        if abs(abs(a) - half_h) <= cap_tol:
            cap_count += 1

    # radial stats (population stddev)
    mean_r = sum(rads) / V
    var_r = sum((r - mean_r) ** 2 for r in rads) / V
    std_r = math.sqrt(var_r)
    rad_std_ratio = std_r / mean_r if mean_r else float('inf')

    # cap ratio (both caps combined)
    cap_ratio = cap_count / V

    # simple height vs radius check (avoid detecting discs as cylinders)
    height = 2.0 * half_h
    radius = mean_r
    height_radius_ok = (radius > 0) and (height / radius >= MIN_HEIGHT_TO_RADIUS)

    is_cylinder = (rad_std_ratio <= RAD_STD_RATIO_THRESH and
                cap_ratio >= (CAP_VERTEX_RATIO * 2) and
                height_radius_ok)

    if is_cylinder:
        return ClothCollisionPrimitiveShape.CYLINDER

    raise RuntimeError(f"Object '{obj.name}' has no cloth primitive type specified in it's name and cannot be deduced from geometry!")


def check_for_bad_lod_suffix(obj: bpy.types.Object):
    """ Checks if the object has an LOD suffix that is known to be ignored by  """

    name = obj.name.lower()
    failure_message = f"Object '{obj.name}' has unknown LOD suffix at the end of it's name!"

    if name.endswith("_lod1"):
        raise RuntimeError(failure_message)
    
    for i in range(4, 10):
        if name.endswith(f"_lod{i}"):
            raise RuntimeError(failure_message)


def select_objects(export_target: str) -> List[bpy.types.Object]:
    """ Returns a list of objects to export. """

    if export_target == "SCENE" or not export_target in {"SELECTED", "SELECTED_WITH_CHILDREN"}:
        return list(bpy.context.scene.objects)

    objects = list(bpy.context.selected_objects)
    added = {obj.name for obj in objects}

    if export_target == "SELECTED_WITH_CHILDREN":
        children = []

        def add_children(parent):
            nonlocal children
            nonlocal added

            for obj in bpy.context.scene.objects:
                if obj.parent == parent and obj.name not in added:
                    children.append(obj)
                    added.add(obj.name)

                    add_children(obj)
        
        for obj in objects:
            add_children(obj)

        objects = objects + children

    parents = []

    for obj in objects:
        parent = obj.parent

        while parent is not None:
            if parent.name not in added:
                parents.append(parent)
                added.add(parent.name)

            parent = parent.parent

    return objects + parents


def expand_armature(armature: bpy.types.Object) -> Dict[str, Model]:

    proper_BONES = get_real_BONES(armature)

    bones: Dict[str, Model] = {}

    for bone in armature.data.bones:
        model = Model()

        transform = bone.matrix_local

        if bone.parent:
            transform = bone.parent.matrix_local.inverted() @ transform
            model.parent = bone.parent.name
        # If the bone has no parent_bone:
        #   set model parent to SKIN object if there is one
        #   set model parent to armature parent if there is one
        else:

            bone_world_matrix = get_bone_world_matrix(armature, bone.name)
            parent_obj = None

            for child_obj in armature.original.children:
                if child_obj.vertex_groups and not get_is_model_hidden(child_obj) and not child_obj.parent_bone:
                    #model.parent = child_obj.name
                    parent_obj = child_obj
                    break

            if parent_obj:
                transform = parent_obj.matrix_world.inverted() @ bone_world_matrix
                model.parent = parent_obj.name
            elif not parent_obj and armature.parent:
                transform = armature.parent.matrix_world.inverted() @ bone_world_matrix 
                model.parent = armature.parent.name
            else:
                transform = bone_world_matrix
                model.parent = ""



        local_translation, local_rotation, _ = transform.decompose()

        model.model_type = ModelType.BONE if bone.name in proper_BONES else ModelType.NULL
        model.name = bone.name
        model.hidden = True
        model.transform.rotation = convert_rotation_space(local_rotation)
        model.transform.translation = convert_vector_space(local_translation)

        bones[bone.name] = model

    return bones
