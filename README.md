# SILICON

**From silicon to pixels.** A from-scratch WebGL2 GPU laboratory — no Three.js, no engine, no magic.

WebGL2 is OpenGL ES 3.00. The objects you create here (`VAO`, `VBO`, `FBO`, `sampler2DShadow`, MRT) are the same objects you would bind from C with GLFW and a desktop 4.x core profile. This repo is a senior-year computer-engineering walk through that pipeline, on your actual GPU.

## Labs

| # | Lab | What you touch |
|---|-----|----------------|
| 00 | Hello Triangle | VAO, interleaved VBO, GLSL ES 3.00, live compile |
| 01 | Coordinate Spaces | Model · View · Projection, column-major `mat4`, orbit / fly camera |
| 02 | Blinn-Phong | Per-pixel lighting, normal matrix, directional + point lights |
| 03 | Sampling | Mipmaps, wrap, anisotropy, tangent-space normal maps |
| 04 | Shadow Maps | Depth FBO, `sampler2DShadow`, PCF, bias |
| 05 | Instancing | `drawElementsInstanced`, divisor-1 matrices, 1 draw for N meshes |
| 06 | G-Buffer | Multiple render targets, deferred geometry buffers |
| 07 | CPU Rasterizer | Barycentric fill, z-buffer, perspective-correct attributes vs the GPU |
| 08 | Wind Tunnel | 3D potential + horseshoe vortex, smoke-rake tracers, swap the body |

## Hardware

On boot the lab calls `glGetString` / `glGetIntegerv` (via WebGL2) and prints the real device:

- unmasked vendor / renderer (`WEBGL_debug_renderer_info`)
- texture / UBO / MRT / MSAA limits
- highp precision, anisotropy, timer-query and float-FBO caps
- the extension string

That panel is the same information a driver engineer reads before trusting a context.

## Architecture

```
src/engine/     math, GL objects, camera, CPU rasterizer
src/labs/       one file per lab, each owns its GPU resources
src/components  lab shell, inspector, live GLSL editor
```

Math is column-major and matches `glUniformMatrix4fv`. Meshes are interleaved `p3 n3 uv2 t4` (48-byte vertices). Nothing in the hot path goes through a scene graph.

## Controls

- **Orbit** (default): drag to orbit, scroll / pinch to dolly
- **Fly**: FPS on-foot. **W/S** forward/back, **A** left, **D** right, **Q/E** lift, Shift sprint. Click the viewport to capture the mouse.
- Keys **0–7** jump labs

## Mapping to desktop OpenGL

| WebGL2 | Desktop |
|--------|---------|
| `createVertexArray` | `glGenVertexArrays` |
| `drawElementsInstanced` | `glDrawElementsInstanced` |
| `framebufferTexture2D` | `glFramebufferTexture2D` |
| `sampler2DShadow` | `sampler2DShadow` + `GL_COMPARE_REF_TO_TEXTURE` |
| `drawBuffers` | `glDrawBuffers` |
| GLSL ES 3.00 | GLSL 3.30+ (layout(location), `in`/`out`) |

## License

MIT
