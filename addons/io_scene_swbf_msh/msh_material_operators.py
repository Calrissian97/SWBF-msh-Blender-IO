""" Operators for basic emulation and mapping of SWBF material system in Blender.
    Only relevant if the builtin Eevee renderer is being used! """

import bpy

from .msh_material_properties import *

from math import sqrt

from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy.types import Operator, Menu

from .option_file_parser import MungeOptions

import os

# FillSWBFMaterialProperties

# Iterates through all material slots of all selected
# objects and fills basic SWBF material properties
# from any Principled BSDF nodes it finds.


class FillSWBFMaterialProperties(bpy.types.Operator):
    bl_idname = "swbf_msh.fill_mat_props"
    bl_label = "Fill SWBF Material Properties"
    bl_description = ("Fill in SWBF properties of all materials used by selected objects.\n"
                "Only considers materials that use nodes.\n" 
                "Please see 'Materials > Materials Operators' in the docs for more details.")

    def execute(self, context):

        slots = sum([list(ob.material_slots) for ob in bpy.context.selected_objects if ob.type == 'MESH'],[])
        mats = [slot.material for slot in slots if (slot.material and slot.material.node_tree)]

        mats_visited = set()

        for mat in mats:

            if mat.name in mats_visited or not mat.swbf_msh_mat:
                continue
            else:
                mats_visited.add(mat.name)

            mat.swbf_msh_mat.doublesided = not mat.use_backface_culling
            mat.swbf_msh_mat.hardedged_transparency = (mat.blend_method == "CLIP")
            mat.swbf_msh_mat.blended_transparency = (mat.blend_method == "BLEND")
            mat.swbf_msh_mat.additive_transparency = (mat.blend_method == "ADDITIVE")

            # Below is all for filling the diffuse map/texture_0 fields

            try:
                for BSDF_node in [n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED']:
                    base_col = BSDF_node.inputs['Base Color'] 

                    stack = []

                    texture_node = None

                    current_socket = base_col
                    if base_col.is_linked:
                        stack.append(base_col.links[0].from_node)

                    while stack:

                        curr_node = stack.pop()

                        if curr_node.type == 'TEX_IMAGE':
                            texture_node = curr_node
                            break
                        else:
                            # Crude but good for now
                            next_nodes = []
                            for node_input in curr_node.inputs:
                                for link in node_input.links:
                                    next_nodes.append(link.from_node)
                            # reversing it so we go from up to down
                            stack += reversed(next_nodes)


                    if texture_node is not None:

                        tex_path = texture_node.image.filepath

                        tex_name = os.path.basename(tex_path)

                        i = tex_name.find('.')
                        
                        # Get rid of trailing number in case one is present
                        if i > 0:
                            tex_name = tex_name[0:i] + ".tga"

                        refined_tex_path = os.path.join(os.path.dirname(tex_path), tex_name)

                        mat.swbf_msh_mat.diffuse_map = refined_tex_path 
                        mat.swbf_msh_mat.texture_0 = refined_tex_path

                        break 
            except:
                # Many chances for null ref exceptions. None if user reads doc section...
                pass  

        return {'FINISHED'}


class VIEW3D_MT_SWBF(bpy.types.Menu):
    bl_label = "SWBF"

    def draw(self, _context):
        layout = self.layout
        layout.operator("swbf_msh.fill_mat_props", text="Fill SWBF Material Properties")


def draw_matfill_menu(self, context):
    layout = self.layout
    layout.separator()
    layout.menu("VIEW3D_MT_SWBF")


def _get_horizontal_cross_cubemap_uv_group():
    """Map a reflection vector into UVs for a 4x3 horizontal cubemap cross."""

    group_name = ".SWBF Horizontal Cross Cubemap UV"
    existing_group = bpy.data.node_groups.get(group_name)
    if existing_group:
        return existing_group

    group = bpy.data.node_groups.new(group_name, "ShaderNodeTree")
    if bpy.app.version < (4, 0, 0):
        group.inputs.new("NodeSocketVector", "Reflection")
        group.outputs.new("NodeSocketVector", "UV")
    else:
        group.interface.new_socket(name="Reflection", in_out='INPUT', socket_type="NodeSocketVector")
        group.interface.new_socket(name="UV", in_out='OUTPUT', socket_type="NodeSocketVector")

    nodes = group.nodes
    links = group.links
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")

    rotate = nodes.new("ShaderNodeVectorRotate")
    rotate.rotation_type = 'X_AXIS'
    rotate.inputs["Angle"].default_value = 1.5707963267948966
    links.new(rotate.inputs["Vector"], group_input.outputs["Reflection"])

    normalize = nodes.new("ShaderNodeVectorMath")
    normalize.operation = 'NORMALIZE'
    links.new(normalize.inputs[0], rotate.outputs["Vector"])
    separate = nodes.new("ShaderNodeSeparateXYZ")
    links.new(separate.inputs["Vector"], normalize.outputs["Vector"])

    def math_node(operation, value_0, value_1=None):
        node = nodes.new("ShaderNodeMath")
        node.operation = operation
        for index, value in enumerate((value_0, value_1)):
            if value is None:
                continue
            if hasattr(value, "bl_idname"):
                links.new(node.inputs[index], value)
            else:
                node.inputs[index].default_value = value
        if operation == 'COMPARE':
            node.inputs[2].default_value = 0.00001
        return node.outputs[0]

    x = separate.outputs["X"]
    y = separate.outputs["Y"]
    z = separate.outputs["Z"]
    abs_x = math_node('ABSOLUTE', x)
    abs_y = math_node('ABSOLUTE', y)
    abs_z = math_node('ABSOLUTE', z)
    max_axis = math_node('MAXIMUM', math_node('MAXIMUM', abs_x, abs_y), abs_z)
    dominant_x = math_node('COMPARE', abs_x, max_axis)
    dominant_y_candidate = math_node('COMPARE', abs_y, max_axis)
    dominant_z_candidate = math_node('COMPARE', abs_z, max_axis)
    not_dominant_x = math_node('SUBTRACT', 1.0, dominant_x)
    dominant_y = math_node('MULTIPLY', dominant_y_candidate, not_dominant_x)
    dominant_z = math_node(
        'MULTIPLY',
        dominant_z_candidate,
        math_node(
            'MULTIPLY',
            not_dominant_x,
            math_node('SUBTRACT', 1.0, dominant_y_candidate)))
    masks = {
        "+X": math_node('MULTIPLY', dominant_x, math_node('GREATER_THAN', x, 0.0)),
        "-X": math_node('MULTIPLY', dominant_x, math_node('LESS_THAN', x, 0.0)),
        "+Y": math_node('MULTIPLY', dominant_y, math_node('GREATER_THAN', y, 0.0)),
        "-Y": math_node('MULTIPLY', dominant_y, math_node('LESS_THAN', y, 0.0)),
        "+Z": math_node('MULTIPLY', dominant_z, math_node('GREATER_THAN', z, 0.0)),
        "-Z": math_node('MULTIPLY', dominant_z, math_node('LESS_THAN', z, 0.0)),
    }
    neg_x = math_node('MULTIPLY', x, -1.0)
    neg_y = math_node('MULTIPLY', y, -1.0)
    neg_z = math_node('MULTIPLY', z, -1.0)

    #       +Y
    # -X +Z +X -Z
    #       -Y
    faces = (
        ("+X", neg_z, neg_y, abs_x, 2.0, 1.0),
        ("-X", z, neg_y, abs_x, 0.0, 1.0),
        ("+Y", x, z, abs_y, 1.0, 0.0),
        ("-Y", x, neg_z, abs_y, 1.0, 2.0),
        ("+Z", x, neg_y, abs_z, 1.0, 1.0),
        ("-Z", neg_x, neg_y, abs_z, 3.0, 1.0),
    )

    face_vectors = []
    for face, horizontal, vertical, major, column, row in faces:
        u = math_node('DIVIDE', horizontal, major)
        u = math_node('ADD', math_node('MULTIPLY', u, 0.5), 0.5 + column)
        u = math_node('MULTIPLY', u, 0.25)
        v = math_node('DIVIDE', vertical, major)
        v = math_node('ADD', math_node('MULTIPLY', v, 0.5), 0.5 + row)
        v = math_node('DIVIDE', v, 3.0)
        combine = nodes.new("ShaderNodeCombineXYZ")
        links.new(combine.inputs["X"], u)
        links.new(combine.inputs["Y"], v)
        masked = nodes.new("ShaderNodeVectorMath")
        masked.operation = 'SCALE'
        links.new(masked.inputs["Vector"], combine.outputs["Vector"])
        links.new(masked.inputs["Scale"], masks[face])
        face_vectors.append(masked.outputs["Vector"])

    cubemap_uv = face_vectors[0]
    for face_vector in face_vectors[1:]:
        add = nodes.new("ShaderNodeVectorMath")
        add.operation = 'ADD'
        links.new(add.inputs[0], cubemap_uv)
        links.new(add.inputs[1], face_vector)
        cubemap_uv = add.outputs["Vector"]
    links.new(group_output.inputs["UV"], cubemap_uv)
    return group


# GenerateMaterialNodesFromSWBFProperties

# Creates shader nodes to emulate SWBF material properties.
# Will probably only support for a narrow subset of properties...
# So much fun to write this, will probably do all render types by end of October

class GenerateMaterialNodesFromSWBFProperties(bpy.types.Operator):
    
    bl_idname = "swbf_msh.generate_material_nodes"
    bl_label = "Generate Nodes"
    bl_description= """Generate Cycles shader nodes from SWBF material properties.
        The nodes generated are meant to give one a general idea
        of how the material would look ingame. They cannot 
        to provide an exact emulation"""

    
    material_name: StringProperty(
        name = "Material Name", 
        description = "Name of material whose SWBF properties the generated nodes will emulate."
    )


    def execute(self, context):

        material = bpy.data.materials.get(self.material_name, None)

        if not material or not material.swbf_msh_mat:
            return {'CANCELLED'}

        mat_props = material.swbf_msh_mat
        is_refraction = "REFRACTION" in mat_props.rendertype
        
        is_emissive = bool(mat_props.glow or mat_props.unlit)

        texture_input_nodes = []
        surface_output_nodes = []

        # Op will give up if no diffuse map is present.
        # Eventually more nuance will be added for different
        # rtypes
        diffuse_texture_path = mat_props.diffuse_map
        if diffuse_texture_path and os.path.exists(diffuse_texture_path):

            material.use_nodes = True
            material.node_tree.nodes.clear()

            bsdf = material.node_tree.nodes.new("ShaderNodeBsdfPrincipled")

            texImage = material.node_tree.nodes.new('ShaderNodeTexImage')
            texImage.image = bpy.data.images.load(diffuse_texture_path)
            texImage.image.alpha_mode = 'CHANNEL_PACKED'
            material.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color']) 

            texture_input_nodes.append(texImage)

            specular_key = "Specular" if bpy.app.version < (4, 0, 0) else "Specular IOR Level"

            bsdf.inputs["Roughness"].default_value = 1.0
            bsdf.inputs[specular_key].default_value = 0.0

            if is_refraction:
                transmission_key = "Transmission" if bpy.app.version < (4, 0, 0) else "Transmission Weight"
                bsdf.inputs[transmission_key].default_value = 1.0
                bsdf.inputs["Roughness"].default_value = 0.0

            material.use_backface_culling = not bool(mat_props.doublesided)

            surface_output_nodes.append(('BSDF', bsdf))

            if not is_emissive:
                if mat_props.hardedged_transparency:
                    material.blend_method = "CLIP"
                    material.node_tree.links.new(bsdf.inputs['Alpha'], texImage.outputs['Alpha'])
                elif mat_props.blended_transparency or mat_props.doublesided or is_refraction:
                    material.blend_method = "BLEND" 
                    material.node_tree.links.new(bsdf.inputs['Alpha'], texImage.outputs['Alpha'])
                elif mat_props.additive_transparency:

                    # most complex 
                    transparent_bsdf = material.node_tree.nodes.new("ShaderNodeBsdfTransparent")
                    add_shader = material.node_tree.nodes.new("ShaderNodeAddShader")

                    material.node_tree.links.new(add_shader.inputs[0], bsdf.outputs["BSDF"])
                    material.node_tree.links.new(add_shader.inputs[1], transparent_bsdf.outputs["BSDF"])

                    surface_output_nodes[0] = ('Shader', add_shader)

            # Glow/Unlit (adds another shader output)
            else:
                emission = material.node_tree.nodes.new("ShaderNodeEmission")
                material.node_tree.links.new(emission.inputs['Color'], texImage.outputs['Color']) 

                emission_strength_multiplier = material.node_tree.nodes.new("ShaderNodeMath")
                emission_strength_multiplier.operation = 'MULTIPLY'
                emission_strength_multiplier.inputs[1].default_value = 32.0

                material.node_tree.links.new(emission_strength_multiplier.inputs[0], texImage.outputs['Alpha']) 
                material.node_tree.links.new(emission.inputs['Strength'], emission_strength_multiplier.outputs[0])

                surface_output_nodes.append(("Emission", emission))

            surfaces_output = None

            if (len(surface_output_nodes) == 1):
                surfaces_output = surface_output_nodes[0][1]
            else:
                mix = material.node_tree.nodes.new("ShaderNodeMixShader")
                material.node_tree.links.new(mix.inputs[1], surface_output_nodes[0][1].outputs[0])
                material.node_tree.links.new(mix.inputs[2], surface_output_nodes[1][1].outputs[0])

                surfaces_output = mix

            # Refraction uses its distortion map as a normal/bump map.
            normal_map_path = mat_props.distortion_map if is_refraction else mat_props.normal_map
            uses_normal_map = "NORMALMAP" in mat_props.rendertype or is_refraction
            if uses_normal_map and normal_map_path and os.path.exists(normal_map_path):
                normalMapTexImage = material.node_tree.nodes.new('ShaderNodeTexImage')
                normalMapTexImage.image = bpy.data.images.load(normal_map_path)
                normalMapTexImage.image.alpha_mode = 'CHANNEL_PACKED'
                normalMapTexImage.image.colorspace_settings.name = 'Non-Color'
                texture_input_nodes.append(normalMapTexImage)

                options = MungeOptions(normal_map_path + ".option")

                if options.get_bool("bumpmap"):

                    # First we must convert the RGB data to brightness
                    rgb_to_bw_node = material.node_tree.nodes.new("ShaderNodeRGBToBW")
                    material.node_tree.links.new(rgb_to_bw_node.inputs["Color"], normalMapTexImage.outputs["Color"])

                    # Now create a bump map node (perhaps we could also use this with normals and just plug color into normal input?)
                    bumpMapNode = material.node_tree.nodes.new('ShaderNodeBump')
                    bumpMapNode.inputs["Distance"].default_value = options.get_float("bumpscale", default=1.0)
                    material.node_tree.links.new(bumpMapNode.inputs["Height"], rgb_to_bw_node.outputs["Val"])

                    normalsOutputNode = bumpMapNode

                else:

                    normalMapNode = material.node_tree.nodes.new('ShaderNodeNormalMap')
                    material.node_tree.links.new(normalMapNode.inputs["Color"], normalMapTexImage.outputs["Color"])

                    normalsOutputNode = normalMapNode
                
                material.node_tree.links.new(bsdf.inputs['Normal'], normalsOutputNode.outputs["Normal"]) 

            elif uses_normal_map:
                map_name = "Distortion map" if is_refraction else "Bumpmap/Normalmap"
                self.report({'WARNING'}, f'{map_name} not found at "{normal_map_path}"!')

            # SWBF gloss maps are stored in texture alpha. Normal/bump alpha
            # takes priority for render types that use such a map.
            specular_mask_output = None
            if bool(mat_props.specular) or "GLOSSMAPPED" in mat_props.rendertype:
                if (
                    "NORMALMAP" in mat_props.rendertype
                    and normal_map_path
                    and os.path.exists(normal_map_path)
                ):
                    specular_mask_output = normalMapTexImage.outputs["Alpha"]
                else:
                    specular_mask_output = texImage.outputs["Alpha"]

                material.node_tree.links.new(
                    bsdf.inputs[specular_key],
                    specular_mask_output)

                gloss_to_roughness = material.node_tree.nodes.new("ShaderNodeMath")
                gloss_to_roughness.label = "Gloss to Roughness"
                gloss_to_roughness.operation = 'SUBTRACT'
                gloss_to_roughness.inputs[0].default_value = 1.0
                material.node_tree.links.new(
                    gloss_to_roughness.inputs[1],
                    specular_mask_output)
                material.node_tree.links.new(
                    bsdf.inputs["Roughness"],
                    gloss_to_roughness.outputs[0])
                specular_tint_input = bsdf.inputs.get("Specular Tint")
                if specular_tint_input is not None:
                    if bpy.app.version < (4, 0, 0):
                        specular_tint_input.default_value = sum(mat_props.specular_color) / 3.0
                    else:
                        specular_tint_input.default_value = (
                            mat_props.specular_color[0],
                            mat_props.specular_color[1],
                            mat_props.specular_color[2],
                            1.0)
                        
            # Environment mapping. SWBF environment maps are traditional cubemaps,
            # so reflection coordinates are used instead of the material UV map.
            uses_environment_map = "ENVMAP" in mat_props.rendertype
            environment_map_path = mat_props.environment_map

            if uses_environment_map and environment_map_path and os.path.exists(environment_map_path):
                reflection_coordinates = material.node_tree.nodes.new("ShaderNodeTexCoord")

                cubemap_coordinates = material.node_tree.nodes.new("ShaderNodeGroup")
                cubemap_coordinates.label = "Horizontal Cross Cubemap UV"
                cubemap_coordinates.node_tree = _get_horizontal_cross_cubemap_uv_group()
                material.node_tree.links.new(
                    cubemap_coordinates.inputs["Reflection"],
                    reflection_coordinates.outputs["Reflection"])

                environment_texture = material.node_tree.nodes.new("ShaderNodeTexImage")
                environment_texture.label = "SWBF Cubemap"
                environment_texture.image = bpy.data.images.load(environment_map_path)
                environment_texture.extension = 'CLIP'
                material.node_tree.links.new(
                    environment_texture.inputs["Vector"],
                    cubemap_coordinates.outputs["UV"])

                environment_tint = material.node_tree.nodes.new("ShaderNodeMixRGB")
                environment_tint.blend_type = 'MULTIPLY'
                environment_tint.inputs["Fac"].default_value = 1.0
                environment_tint.inputs["Color2"].default_value = (
                    mat_props.specular_color[0],
                    mat_props.specular_color[1],
                    mat_props.specular_color[2],
                    1.0)
                material.node_tree.links.new(
                    environment_tint.inputs["Color1"],
                    environment_texture.outputs["Color"])

                environment_emission = material.node_tree.nodes.new("ShaderNodeEmission")
                material.node_tree.links.new(
                    environment_emission.inputs["Color"],
                    environment_tint.outputs["Color"])
                if specular_mask_output is not None:
                    material.node_tree.links.new(
                        environment_emission.inputs["Strength"],
                        specular_mask_output)

                add_environment = material.node_tree.nodes.new("ShaderNodeAddShader")
                material.node_tree.links.new(
                    add_environment.inputs[0],
                    surfaces_output.outputs[0])
                material.node_tree.links.new(
                    add_environment.inputs[1],
                    environment_emission.outputs["Emission"])
                surfaces_output = add_environment

            elif uses_environment_map:
                self.report(
                    {'WARNING'},
                    f'Environment map/cubemap not found at "{environment_map_path}"!')
            output = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
            material.node_tree.links.new(output.inputs['Surface'], surfaces_output.outputs[0]) 

            # Scrolling
            # This approach works 90% of the time, but notably produces very incorrect results
            # on mus1_bldg_world_1,2,3

            # Clear all anims in all cases
            if material.node_tree.animation_data:
                material.node_tree.animation_data_clear()

            if "BLINK" in mat_props.rendertype:
                blink_brightness = material.node_tree.nodes.new("ShaderNodeValue")
                blink_brightness.label = "SWBF Blink Brightness"

                blink_multiply = material.node_tree.nodes.new("ShaderNodeMixRGB")
                blink_multiply.label = "SWBF Blink"
                blink_multiply.blend_type = 'MULTIPLY'
                blink_multiply.inputs["Fac"].default_value = 1.0

                for link in list(bsdf.inputs["Base Color"].links):
                    material.node_tree.links.remove(link)

                material.node_tree.links.new(
                    blink_multiply.inputs["Color1"],
                    texImage.outputs["Color"])
                material.node_tree.links.new(
                    blink_multiply.inputs["Color2"],
                    blink_brightness.outputs["Value"])
                material.node_tree.links.new(
                    bsdf.inputs["Base Color"],
                    blink_multiply.outputs["Color"])

                if is_emissive:
                    for link in list(emission.inputs["Color"].links):
                        material.node_tree.links.remove(link)
                    material.node_tree.links.new(
                        emission.inputs["Color"],
                        blink_multiply.outputs["Color"])
                minimum_brightness = max(
                    0.0, min(1.0, mat_props.blink_min_brightness / 255.0))
                blink_brightness.outputs["Value"].default_value = 1.0

                if mat_props.blink_speed > 0 and minimum_brightness < 1.0:
                    radians_per_frame = (
                        mat_props.blink_speed * 0.5
                        / max(1.0, context.scene.render.fps))
                    brightness_midpoint = (1.0 + minimum_brightness) * 0.5
                    brightness_amplitude = (1.0 - minimum_brightness) * 0.5

                    blink_fcurve = blink_brightness.outputs["Value"].driver_add(
                        "default_value")
                    blink_driver = blink_fcurve.driver
                    blink_driver.type = 'SCRIPTED'

                    frame_variable = blink_driver.variables.new()
                    frame_variable.name = "frame"
                    frame_variable.type = 'SINGLE_PROP'
                    frame_variable.targets[0].id_type = 'SCENE'
                    frame_variable.targets[0].id = context.scene
                    frame_variable.targets[0].data_path = "frame_current"

                    blink_driver.expression = (
                        f"{brightness_midpoint:.9g}"
                        f" + {brightness_amplitude:.9g}"
                        f" * sin(frame * {radians_per_frame:.9g})"
                    )

            if "SCROLL" in mat_props.rendertype:
                uv_input = material.node_tree.nodes.new("ShaderNodeUVMap")

                vector_add = material.node_tree.nodes.new("ShaderNodeVectorMath")

                # Add keyframes
                scroll_per_sec_divisor = 255.0 
                frame_step = 60.0
                fps = bpy.context.scene.render.fps
                for i in range(2):
                    vector_add.inputs[1].default_value[0] = i * mat_props.scroll_speed_u * frame_step / scroll_per_sec_divisor              
                    vector_add.inputs[1].keyframe_insert("default_value", index=0, frame=i * frame_step * fps)

                    vector_add.inputs[1].default_value[1] = i * mat_props.scroll_speed_v * frame_step / scroll_per_sec_divisor               
                    vector_add.inputs[1].keyframe_insert("default_value", index=1, frame=i * frame_step * fps)

                material.node_tree.links.new(vector_add.inputs[0], uv_input.outputs[0])

                for texture_node in texture_input_nodes:
                    material.node_tree.links.new(texture_node.inputs["Vector"], vector_add.outputs[0])

            # Don't know how to set interpolation when adding keyframes
            # so we must do it after the fact
            if material.node_tree.animation_data and material.node_tree.animation_data.action:
                action = material.node_tree.animation_data.action
                # Blender 4.x
                if hasattr(action, "fcurves"):
                    for fcurve in action.fcurves:
                        for kf in fcurve.keyframe_points:
                            kf.interpolation = 'LINEAR'
                # Blender 5.0+
                else:
                    for layer in action.layers:
                        for strip in layer.strips:
                            for channelbag in strip.channelbags:
                                for fcurve in channelbag.fcurves:
                                    for kf in fcurve.keyframe_points:
                                        kf.interpolation = 'LINEAR'

        else:
            self.report({'WARNING'}, f"Diffuse texture not found at {diffuse_texture_path}!")

        return {'FINISHED'}
