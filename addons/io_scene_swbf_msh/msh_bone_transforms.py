"""New module for making it a bit easier to work with SWBF skeletons in Blender.
This adds non-bone objects in the skeleton hierarchy to their own collection,
as well as creates a neat Head->Tails hierarchy of bones typical of a Blender
skeleton/armature.
"""

from mathutils import Matrix

def is_skeleton_helper(name):
    """Identify skelly helpers from actual bones in the SWBF skelly"""
    NodeName = str(name).lower()
    if NodeName.startswith("root_") or NodeName.startswith("eff_") or NodeName.startswith("hp_"):
        return True
    else:
        return False
    

def create_bone_collections(armature):
    """Organize the SWBF skelly into collections of bones and of helpers"""
    ActualBones = armature.collections.new("SWBF Bones")
    SkeletonHelpers = armature.collections.new("SWBF Skeleton Helpers")
    for bone in armature.bones:
        if is_skeleton_helper(bone.name):
            SkeletonHelpers.assign(bone)
            bone.use_deform = False         # Let's also go ahead and prevent it from influencing vertices
        else:
            ActualBones.assign(bone)
    SkeletonHelpers.is_visible = False      # Have skelly helpers hidden by default (they shouldn't be edited often, if at all)


def get_swbf_bone_axis_offset(bone):
    """Retrieve the SWBF MSH axis offset for this bone and return as matrix,
    or simply return an identity matrix for multiplying later."""
    values = bone.get("swbf_msh_axis_offset")                   # This property holds the original bone's transforms
    if values is None:
        return Matrix.Identity(4)                               # Otherwise return identity matrix if property is missing
    return Matrix([values[i:i + 4] for i in range(0, 16, 4)])   # Return original transforms as matrix


def calculate_swbf_bone_matrix(bone):
    """Multiply a bone's Blender transforms matrix by the bone's stored offset to get the original SWBF matrix."""
    return bone.matrix_local @ get_swbf_bone_axis_offset(bone)


def create_bone_hierarchy(armature):
    """Creates a clean hierarchy of Blender bone objects with tails pointing towards
    children for a neater and easily editable appearance. Also creates the
    swbf_msh_axis_offset bone property from original matrix"""

    originals = {b.name: b.matrix.copy() for b in armature.edit_bones}  # Store original matrices
    for bone in armature.edit_bones:
        if is_skeleton_helper(bone.name):   # Skip helpers
            continue

        # First let's get the directions bone tails should point towards
        directions = [originals[c.name].translation - bone.head for c in bone.children] # Get difference of child(ren) from parent bone
        directions = [direction for direction in directions if direction.length > 1e-6] # Filter out insignificant differences

        if directions:
            # Try to aim bone tails at average direction of child bone(s)
            direction = sum(directions[1:], directions[0].copy()) / len(directions)

            if direction.length < 1e-6:
                direction = directions[0]   # Just point at first child if average is insignificant
            bone.tail = bone.head + direction
            
        elif bone.parent:
            ParentBone = bone.parent

            # Travel up hierarchy to first actual bone parent
            while ParentBone.parent and is_skeleton_helper(ParentBone.name):
                ParentBone = ParentBone.parent

            # If no children, calculate from parent head instead
            direction = bone.head - ParentBone.head
            if direction.length > 1e-6:
                bone.tail = bone.head + direction * 0.3

        # Have original Z axis the rolling axis
        bone.align_roll(originals[bone.name].to_3x3().col[2])

        # Calculate offset from inverse blender matrix multiplied by original matrix
        offset = bone.matrix.inverted() @ originals[bone.name]
        offset.translation = (0, 0, 0)  # We're only storing rotation differences

        # Assign this bone's MSH offset property
        bone["swbf_msh_axis_offset"] = [v for row in offset for v in row]
