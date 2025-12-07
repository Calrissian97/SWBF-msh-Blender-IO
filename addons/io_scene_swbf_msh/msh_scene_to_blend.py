""" Gathers the Blender objects from the current scene and returns them as a list of
    Model objects. """

import os
import bpy
import bmesh
import math
from enum import Enum
from typing import List, Set, Dict, Tuple
from .msh_scene import Scene
from .msh_material_to_blend import *
from .msh_model import *
from .msh_skeleton_utilities import *
from .msh_skeleton_to_blend import *
from .msh_model_gather import get_is_model_hidden
from .msh_mesh_to_blend import model_to_mesh_object
from .msh_model_utilities import convert_vector_space, convert_rotation_space
from .crc import *

def apply_transform(obj: bpy.types.Object, transform: ModelTransform):
    """ Applies the msh transform to a blender object. """
    obj.location = convert_vector_space(transform.translation)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = convert_rotation_space(transform.rotation)


# Create the msh hierachy.  Armatures are not created here.
def extract_models(scene: Scene, materials_map : Dict[str, bpy.types.Material]) -> Dict[str, bpy.types.Object]:

    # This will be filled with model names -> Blender objects and returned
    model_map : Dict[str, bpy.types.Object] = {}

    sorted_models : List[Model] = sort_by_parent(scene.models)

    for model in sorted_models:
        
        new_obj = None

        # Handle cloth models first
        if model.cloth:
            cloth = model.cloth
            mesh_data = bpy.data.meshes.new(name=cloth.name)
            
            positions = [convert_vector_space(pos) for pos in cloth.positions]
            # Populate mesh with cloth data
            mesh_data.from_pydata(positions, [], cloth.triangles)

            # Add UV map if present
            if cloth.uvs:
                uv_layer = mesh_data.uv_layers.new(name="UVMap")
                for poly in mesh_data.polygons:
                    for loop_index in poly.loop_indices:
                        loop = mesh_data.loops[loop_index]
                        vert_index = loop.vertex_index
                        if vert_index < len(cloth.uvs):
                            uv_layer.data[loop_index].uv = cloth.uvs[vert_index]

            mesh_data.update()
            mesh_data.validate()

            # Create object
            new_obj = bpy.data.objects.new(cloth.name, mesh_data)

            # Store cloth collision primitives as custom prop
            if cloth.collision_objects:
                new_obj["swbf_msh_cloth_collisions"] = str([coll.name for coll in cloth.collision_objects])
            
            # Store original cloth constraints data as custom properties for export
            new_obj["swbf_msh_cloth_stretch_constraints"] = str(cloth.stretch_constraints)
            new_obj["swbf_msh_cloth_cross_constraints"] = str(cloth.cross_constraints)
            new_obj["swbf_msh_cloth_bend_constraints"] = str(cloth.bend_constraints)

            # Store a signature of the mesh to detect edits on export
            mesh_signature = f"{len(mesh_data.vertices)}:{len(mesh_data.edges)}:{len(mesh_data.polygons)}"
            new_obj["swbf_msh_cloth_mesh_signature"] = mesh_signature

            # Add fixed points as a "Pin" vertex group
            if cloth.fixed_points:
                pin_group = new_obj.vertex_groups.new(name="Pin")
                pin_group.add(cloth.fixed_points, 1.0, 'REPLACE')

            # If there are bones associated with the fixed points, assign vertex weights
            # so the cloth deforms with the armature.
            if cloth.fixed_weights_bones and len(cloth.fixed_weights_bones) == len(cloth.fixed_points):
                for i, bone_name in enumerate(cloth.fixed_weights_bones):
                    vertex_index = cloth.fixed_points[i]

                    # Get or create a vertex group for the bone
                    bone_vertex_group = new_obj.vertex_groups.get(bone_name)
                    if not bone_vertex_group:
                        bone_vertex_group = new_obj.vertex_groups.new(name=bone_name)

                    # Assign the vertex to the bone's group with a full weight of 1.0
                    bone_vertex_group.add([vertex_index], 1.0, 'ADD')

            # Create and assign material
            if cloth.texture:
                # Create a temporary msh.Material to pass to the material builder
                cloth_mat_name = os.path.splitext(cloth.texture)[0]
                blender_material = bpy.data.materials.get(cloth_mat_name)

                if not blender_material:
                    # Create a new Blender material and an msh.Material to populate its properties
                    blender_material = bpy.data.materials.new(name=cloth_mat_name)
                    cloth_msh_material = Material(name=cloth_mat_name, texture0=cloth.texture)
                    fill_material_props(cloth_msh_material, blender_material.swbf_msh_mat, "")

                # Disable backface culling for cloth materials so they are double-sided
                blender_material.use_backface_culling = False

                new_obj.data.materials.append(blender_material)

        elif model.geometry:
            new_obj = model_to_mesh_object(model, scene, materials_map)

        else:
            new_obj = bpy.data.objects.new(model.name, None)
            new_obj.empty_display_size = 1
            new_obj.empty_display_type = 'PLAIN_AXES' 


        model_map[model.name] = new_obj
        new_obj.name = model.name

        if model.parent:
            new_obj.parent = model_map[model.parent]

        apply_transform(new_obj, model.transform)
        if model.collisionprimitive is not None:
            new_obj.swbf_msh_coll_prim.prim_type = model.collisionprimitive.shape.value

        bpy.context.collection.objects.link(new_obj)


    return model_map


# TODO: Add to custom material info struct, maybe some material conversion/import?
def extract_materials(folder_path: str, scene: Scene) -> Dict[str, bpy.types.Material]:

    extracted_materials : Dict[str, bpy.types.Material] = {}

    for material_name, material in scene.materials.items():

        extracted_materials[material_name] = swbf_material_to_blend(material_name, material, folder_path)

    return extracted_materials


#TODO: Examine cloth collision parenting
def extract_scene(filepath: str, scene: Scene):

    folder = os.path.join(os.path.dirname(filepath),"")

    # material_map mapes Material names to Blender materials
    material_map = extract_materials(folder, scene)

    # model_map maps Model names to Blender objects.
    model_map = extract_models(scene, material_map)


    # skel contains all models needed in an armature
    skel = extract_required_skeleton(scene)

    # Create the armature if skel is non-empty
    armature = None if not skel else required_skeleton_to_armature(skel, model_map, scene)

    if armature is not None:
        preserved_skel = armature.data.swbf_msh_skel
        for model in scene.models:
            if to_crc(model.name) in scene.skeleton or model.model_type == ModelType.BONE:
                entry = preserved_skel.add()
                entry.name = model.name


    '''
    If an armature was created, we need to do a few extra 
    things to ensure the import makes sense in Blender.  It can
    get a bit messy, as XSI + SWBF have very loose requirements  
    when it comes to skin-skeleton parentage.

    If not, we're good.
    '''
    if armature is not None:

        has_skin = False

        # Handle armature related parenting
        for curr_model in scene.models:

            curr_obj = model_map[curr_model.name]
            
            # Parent all skins and cloth to the armature
            if curr_model.model_type in (ModelType.SKIN, ModelType.CLOTH):

                has_skin = True

                worldmat = curr_obj.matrix_world
                curr_obj.parent = armature
                curr_obj.parent_type = 'ARMATURE'
                curr_obj.matrix_world = worldmat

            # Parent the object to a bone if necessary
            else:
                parent_bone_name = ""
                if curr_model.name in armature.data.bones and curr_model.geometry:
                    parent_bone_name = curr_model.name
                elif curr_model.parent in armature.data.bones and curr_model.name not in armature.data.bones:
                    parent_bone_name = curr_model.parent

                if parent_bone_name:
                    # Not sure what the different mats do, but saving the worldmat and 
                    # applying it after clearing the other mats yields correct results...
                    worldmat = curr_obj.matrix_world

                    curr_obj.parent = armature
                    curr_obj.parent_type = 'BONE'
                    curr_obj.parent_bone = parent_bone_name
                    # ''
                    curr_obj.matrix_basis = Matrix()
                    curr_obj.matrix_parent_inverse = Matrix()
                    curr_obj.matrix_world = worldmat


        '''
        Sometimes skins are parented to other skins.  We need to find the skin highest in the hierarchy and
        parent all skins to its parent (armature_reparent_obj).

        If not skin exists, we just reparent the armature to the parent of the highest node in the skeleton
        '''
        armature_reparent_obj = None
        if has_skin:
            for model in sort_by_parent(scene.models):
                if model.model_type == ModelType.SKIN:
                    armature_reparent_obj = None if not model.parent else model_map[model.parent]
        else:
            skeleton_parent_name = skel[0].parent
            for model in scene.models:
                if model.name == skeleton_parent_name:
                    armature_reparent_obj = None if not skeleton_parent_name else model_map[skeleton_parent_name]

        # Now we reparent the armature to the node (armature_reparent_obj) we just found
        if armature_reparent_obj is not None and armature.name != armature_reparent_obj.name:
            world_tx = armature.matrix_world
            armature.parent = armature_reparent_obj
            armature.matrix_basis = Matrix()
            armature.matrix_parent_inverse = Matrix()
            armature.matrix_world = Matrix.Identity(4)


        # If an bone exists in the armature, delete its 
        # object counterpart (as created in extract_models)
        for bone in skel:
            model_to_remove = model_map[bone.name]
            if model_to_remove and model_to_remove.parent_bone == "":
                bpy.data.objects.remove(model_to_remove, do_unlink=True)
                model_map.pop(bone.name)

        armature.matrix_world = Matrix.Identity(4)        
    

    
    for model in scene.models:
        # Lastly, hide all that is hidden in the msh scene
        if model.name in model_map:
            obj = model_map[model.name]
            obj.hide_set(model.hidden or get_is_model_hidden(obj))
