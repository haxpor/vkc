# vkc

vkc aims to be more like a helper Vulkan library to ease integration Vulkan graphics into
your game. It aims to be tightly integrated with Blender 2.8+ both in term of file format exports,
and using Blender as scene editor importing into the game.

This is work-in-progress.

# Build

* `make` - to build `libvkc.so`, and testing program
* `make test` - to build testing program consuming the library `libvkc.so`.
* `make clean` - to clean the built files

# Execution

Prior to running the test program, make sure you download the asset binary file from https://github.com/haxpor/vkc/releases then extract it at the root level of the project, then run the program.

In case you checkout an older commit, you have to make sure to use not newer resource. Check its commit hash.

# TODO

* [x] Initial foundation code to render something on screen
* [ ] Deferred shading
* [ ] Skybox
* [ ] Environmental cubemap
* [ ] Text rendering
* [ ] Soft shadow (possibly Percentage-Closer Soft Shadow technique)
* [ ] Support importing `.obj`, `.gltf` (2.0) (former aim for modeling, later for all included animation and scene setup)
* [ ] Easing function integrated into suitable sub-systems
* [ ] SSAO - Screen-space ambient occlusion
* [ ] TAA - Temporal Anti-aliasing
* [ ] Albedo, Normal, Specular, Emissive (might consider PBR)
* [ ] Bounding box for mesh/model
* [ ] Ray, and Plane collision detection
* [ ] Performance statistics text rendering on screen
* [ ] View frustum culling
* [ ] Terrain generation/rendering
* [ ] Tessellation for terrain LOD
* [ ] Support textures in format PNG, TGA, DDS (with DXT/BC)
* [ ] Directional light, Spot light, and Omni-light
* [ ] Reflection, and Refraction plane
* [ ] `.mtl` material file support
* maybe others...
