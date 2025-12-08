# SWBF-msh-Blender-IO
.msh (SWBF toolchain version) Importer-Exporter for Blender

### Changes from Original
This fork of PrismaticFlower's addon adds support for importing/exporting cloth, see the [Cloth section](https://github.com/Calrissian97/SWBF-msh-Blender-IO/blob/master/docs/reference_manual.md#cloth) of the reference manual for details.

### Installing
You install it like any other Blender addon, if you already know how to do that then great! Else head over [here](https://docs.blender.org/manual/en/3.0/editors/preferences/addons.html#installing-add-ons) to learn how to do it in Blender 3.0.

### Reference Manual
Included in the repository is a [Reference Manual](https://github.com/Calrissian97/SWBF-msh-Blender-IO/blob/master/docs/reference_manual.md#reference-manual) of sorts. There is no need to read through it before using the addon but anytime you have a question about how something works or why an export failed it should hopefully have the answers.

### What from [glTF-Blender-IO](https://github.com/KhronosGroup/glTF-Blender-IO) was used?
The `reload_package` function from \_\_init\_\_.py. Before writing this I had barely touched Python and when I saw that glTF-Blender-IO had a function to assist script reloading "I thought that's useful, I think I kinda need that and I don't know how to write something like that myself yet.". And it was very useful, so thank you to all the glTF-Blender-IO developers and contributors.
