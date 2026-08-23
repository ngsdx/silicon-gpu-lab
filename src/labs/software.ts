import { drawMesh } from "@/engine/buffer";
import { createProgram } from "@/engine/shader";
import { disposeCommon, makeFullscreenTri, setCameraUniforms, uploadMesh, VS_LIT, FS_UNLIT } from "@/engine/common-gl";
import { createIcosphere } from "@/engine/mesh";
import { SoftwareRaster } from "@/engine/software-raster";
import { createTexture2D } from "@/engine/texture";
import { mat4FromRotationY, mat4Identity, mat4Multiply, mat4Normal } from "@/engine/math";
import type { Camera } from "@/engine/camera";
import type { LabDefinition, LabInstance } from "@/engine/lab-types";
import type { GlContext } from "@/engine/gpu";

const VS_BLIT = `#version 300 es
layout(location=0) in vec2 aPos; out vec2 vUv;
void main(){ vUv = vec2(aPos.x*0.5+0.5, aPos.y*0.5+0.5); gl_Position = vec4(aPos,0.0,1.0); }`;
const FS_BLIT = `#version 300 es
precision highp float;
in vec2 vUv; uniform sampler2D uTex; out vec4 frag;
void main(){ frag = texture(uTex, vUv); }`;

export const softwareLab: LabDefinition = {
  id: "software",
  index: "07",
  title: "CPU Rasterizer",
  subtitle: "Barycentric · z-buffer · vs GPU",
  pipeline: ["VS (CPU)", "Perspective divide", "Barycentric", "Z-test", "ROP"],
  params: [
    { key: "res", label: "CPU width", min: 80, max: 320, step: 16, default: 160 },
    { key: "shade", label: "Shade (0 bary, 1 nrm, 2 z)", min: 0, max: 2, step: 1, default: 0 },
    { key: "spin", label: "Spin", min: 0, max: 2, step: 0.01, default: 0.5 },
  ],
  note: {
    title: "What the ROP actually does",
    body: "This lab runs the rasterizer on the CPU: clip-space transform, perspective divide, viewport, edge-function barycentrics, a top-left fill test, perspective-correct attributes, a z-buffer, then a write to a color buffer. The left half is the GPU doing the same mesh in one draw call. The right half is your core doing it in JS. The millisecond gap is the reason GPUs exist — SIMD, hierarchical z, and dedicated ROP hardware.",
    glsl: "// CPU equivalent of the rasterizer\nfloat w0 = edge(b, c, p), w1 = edge(c, a, p), w2 = edge(a, b, p);\nif (w0>=0 && w1>=0 && w2>=0) { /* inside */ }",
    mapping: "GPU: rasterizer + ROP fixed-function\nCPU: nested loops over the bounding box\ngl_FragCoord.z ↔ ndc.z * 0.5 + 0.5",
  },
  create(ctx: GlContext): LabInstance {
    const { gl } = ctx;
    const cpuMesh = createIcosphere(1, 1);
    const gpuMesh = uploadMesh(gl, cpuMesh);
    const prog = createProgram(gl, VS_LIT, FS_UNLIT);
    const blit = createProgram(gl, VS_BLIT, FS_BLIT);
    const tri = makeFullscreenTri(gl);
    let raster = new SoftwareRaster(160, 120);
    let tex = createTexture2D(gl, raster.width, raster.height, raster.color, {
      wrap: gl.CLAMP_TO_EDGE,
      mag: gl.NEAREST,
      min: gl.NEAREST,
      mipmap: false,
    });
    const model = mat4Identity();
    const mvp = mat4Identity();
    const tmp = mat4Identity();
    const normal = mat4Identity();
    let cam: Camera | null = null;
    let t = 0, res = 160, shade = 0, spin = 0.5;

    const shades = ["bary", "normal", "depth"] as const;

    return {
      update(dt, _i, params, camera) {
        t += dt;
        cam = camera;
        res = params.res ?? 160;
        shade = params.shade ?? 0;
        spin = params.spin ?? 0.5;
      },
      draw(w, h) {
        if (!cam) return { draws: 0, tris: 0, instances: 0, gpuMs: null, cpuMs: 0 };
        const rw = Math.max(64, res | 0);
        const rh = Math.max(48, Math.round(rw * 0.75));
        if (rw !== raster.width || rh !== raster.height) {
          raster = new SoftwareRaster(rw, rh);
          gl.deleteTexture(tex);
          tex = createTexture2D(gl, rw, rh, raster.color, {
            wrap: gl.CLAMP_TO_EDGE,
            mag: gl.NEAREST,
            min: gl.NEAREST,
            mipmap: false,
          });
        }

        mat4FromRotationY(model, t * spin);
        mat4Multiply(tmp, cam.view, model);
        mat4Multiply(mvp, cam.proj, tmp);

        raster.clear();
        raster.drawIndexed(
          cpuMesh.vertices,
          12,
          cpuMesh.indices,
          mvp,
          shades[Math.max(0, Math.min(2, shade | 0))]!,
        );
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, rw, rh, gl.RGBA, gl.UNSIGNED_BYTE, raster.color);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);

        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.viewport(0, 0, Math.floor(w / 2), h);
        gl.useProgram(prog.prog);
        setCameraUniforms(gl, prog, cam, model);
        gl.uniformMatrix4fv(prog.uniforms.uNormal, false, mat4Normal(normal, model));
        gl.uniform3f(prog.uniforms.uColor, 0.82, 0.32, 0.28);
        drawMesh(gl, gpuMesh);

        gl.disable(gl.DEPTH_TEST);
        gl.viewport(Math.floor(w / 2), 0, Math.floor(w / 2), h);
        gl.useProgram(blit.prog);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.uniform1i(blit.uniforms.uTex, 0);
        drawMesh(gl, tri);
        gl.enable(gl.DEPTH_TEST);
        gl.viewport(0, 0, w, h);

        return {
          draws: 2,
          tris: cpuMesh.triCount,
          instances: 1,
          gpuMs: null,
          cpuMs: raster.ms,
          extra: {
            "CPU fill": raster.pixelsFilled,
            "CPU tested": raster.pixelsTested,
            "CPU ms": raster.ms.toFixed(1),
            res: `${rw}×${rh}`,
          },
        };
      },
      dispose() {
        gl.deleteTexture(tex);
        disposeCommon(gl, [gpuMesh, tri], [prog, blit]);
      },
    };
  },
};
