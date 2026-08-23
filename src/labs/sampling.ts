import { drawMesh } from "@/engine/buffer";
import { createProgram } from "@/engine/shader";
import { disposeCommon, makeGrid, makeLineProgram, drawGrid, setCameraUniforms, uploadMesh, VS_LIT } from "@/engine/common-gl";
import { createCube, createPlane } from "@/engine/mesh";
import {
  bindTexture,
  createTexture2D,
  heightToNormal,
  makeBrickAlbedo,
  makeBrickHeight,
  makeChecker,
  makeMarble,
} from "@/engine/texture";
import { mat4FromTranslation, mat4Identity, mat4Normal } from "@/engine/math";
import type { Camera } from "@/engine/camera";
import type { LabDefinition, LabInstance } from "@/engine/lab-types";
import type { GlContext } from "@/engine/gpu";

const FS_TEX = `#version 300 es
precision highp float;
in vec3 vWorld;
in vec3 vNrm;
in vec2 vUv;
in mat3 vTBN;
uniform sampler2D uAlbedo;
uniform sampler2D uNormalMap;
uniform vec3 uEye;
uniform float uTiling;
uniform float uUseNormal;
uniform float uSpec;
out vec4 frag;
void main(){
  vec2 uv = vUv * uTiling;
  vec3 albedo = texture(uAlbedo, uv).rgb;
  vec3 n;
  if (uUseNormal > 0.5) {
    vec3 nt = texture(uNormalMap, uv).xyz * 2.0 - 1.0;
    n = normalize(vTBN * nt);
  } else {
    n = normalize(vNrm);
  }
  vec3 L = normalize(vec3(0.45, 0.8, 0.35));
  vec3 V = normalize(uEye - vWorld);
  vec3 H = normalize(L + V);
  float diff = max(dot(n, L), 0.0);
  float spec = pow(max(dot(n, H), 0.0), 48.0) * uSpec;
  vec3 col = albedo * (0.12 + 0.88 * diff) + spec * vec3(1.0);
  frag = vec4(col, 1.0);
}`;

export const samplingLab: LabDefinition = {
  id: "sampling",
  index: "03",
  title: "Sampling",
  subtitle: "Mipmaps · wrap · anisotropy · TBN",
  pipeline: ["UV", "Filter", "Aniso", "Normal map", "Lighting"],
  params: [
    { key: "tiling", label: "UV tiling", min: 1, max: 24, step: 1, default: 6 },
    { key: "map", label: "Map (0 check, 1 marble, 2 brick)", min: 0, max: 2, step: 1, default: 2 },
    { key: "normal", label: "Normal map", min: 0, max: 1, step: 1, default: 1 },
    { key: "aniso", label: "Anisotropy", min: 1, max: 16, step: 1, default: 8 },
    { key: "spec", label: "Specular", min: 0, max: 1, step: 0.01, default: 0.35 },
  ],
  note: {
    title: "The texture unit is a hardware sampler",
    body: "A texture is not an image — it is a bound object with wrap, min/mag filters, a mip chain, and an anisotropic tap budget. Mipmaps exist because a distant polygon covers less than a texel; trilinear + anisotropy is how GPUs keep minification stable at grazing angles. The brick path also builds a tangent-space normal map from a height field (finite differences) and transforms it by the TBN matrix uploaded per vertex.",
    glsl: "vec3 nt = texture(uNormalMap, uv).xyz * 2.0 - 1.0;\nvec3 N = normalize(TBN * nt);",
    mapping: "glTexParameteri(GL_TEXTURE_MAX_ANISOTROPY_EXT)\nglGenerateMipmap(GL_TEXTURE_2D)\nsampler2D is a uniform pointing at a texture unit",
  },
  create(ctx: GlContext): LabInstance {
    const { gl } = ctx;
    const cube = uploadMesh(gl, createCube(1.4));
    const plane = uploadMesh(gl, createPlane(16, 1, 0));
    const grid = makeGrid(gl);
    const lines = makeLineProgram(gl);
    const prog = createProgram(gl, VS_LIT, FS_TEX);
    const checker = createTexture2D(gl, 256, 256, makeChecker(256, 8));
    const marble = createTexture2D(gl, 256, 256, makeMarble(256));
    const brick = createTexture2D(gl, 256, 256, makeBrickAlbedo(256));
    const nrm = createTexture2D(gl, 256, 256, heightToNormal(makeBrickHeight(256), 256, 6));
    const maps = [checker, marble, brick];
    const model = mat4Identity();
    const normalM = mat4Identity();
    let cam: Camera | null = null;
    let tiling = 6, map = 2, useN = 1, aniso = 8, spec = 0.35;

    const applyAniso = (tex: WebGLTexture) => {
      const ext = ctx.anisotropyExt;
      if (!ext) return;
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameterf(gl.TEXTURE_2D, ext.TEXTURE_MAX_ANISOTROPY_EXT, aniso);
    };

    return {
      update(_dt, _i, params, camera) {
        cam = camera;
        tiling = params.tiling ?? 6;
        map = params.map ?? 2;
        useN = params.normal ?? 1;
        aniso = params.aniso ?? 8;
        spec = params.spec ?? 0.35;
      },
      draw() {
        if (!cam) return { draws: 0, tris: 0, instances: 0, gpuMs: null, cpuMs: 0 };
        const albedo = maps[Math.max(0, Math.min(2, map | 0))]!;
        applyAniso(albedo);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        drawGrid(gl, grid, lines, cam);
        gl.useProgram(prog.prog);
        gl.uniform1f(prog.uniforms.uTiling, tiling);
        gl.uniform1f(prog.uniforms.uUseNormal, map === 2 ? useN : 0);
        gl.uniform1f(prog.uniforms.uSpec, spec);
        gl.uniform3fv(prog.uniforms.uEye, cam.eye);
        bindTexture(gl, 0, albedo);
        bindTexture(gl, 1, nrm);
        gl.uniform1i(prog.uniforms.uAlbedo, 0);
        gl.uniform1i(prog.uniforms.uNormalMap, 1);

        mat4FromTranslation(model, [0, 0.7, 0]);
        setCameraUniforms(gl, prog, cam, model);
        gl.uniformMatrix4fv(prog.uniforms.uNormal, false, mat4Normal(normalM, model));
        drawMesh(gl, cube);

        mat4Identity(model);
        setCameraUniforms(gl, prog, cam, model);
        gl.uniformMatrix4fv(prog.uniforms.uNormal, false, mat4Normal(normalM, model));
        drawMesh(gl, plane);

        return {
          draws: 3,
          tris: 14,
          instances: 2,
          gpuMs: null,
          cpuMs: 0,
          extra: { aniso: `${aniso}x`, mips: "on" },
        };
      },
      dispose() {
        disposeCommon(gl, [cube, plane, grid], [prog, lines]);
        for (const t of [...maps, nrm]) gl.deleteTexture(t);
      },
    };
  },
};
