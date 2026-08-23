import { drawMesh } from "@/engine/buffer";
import { createProgram, destroyProgram, tryCreateProgram } from "@/engine/shader";
import {
  disposeCommon,
  drawGrid,
  makeGrid,
  makeLineProgram,
  setCameraUniforms,
  uploadMesh,
  VS_LIT,
} from "@/engine/common-gl";
import { createCube, createIcosphere, createPlane, createTorus } from "@/engine/mesh";
import { mat4FromRotationY, mat4FromTranslation, mat4Identity, mat4Multiply, mat4Normal } from "@/engine/math";
import type { Camera } from "@/engine/camera";
import type { LabDefinition, LabInstance } from "@/engine/lab-types";
import type { GlContext } from "@/engine/gpu";

export const VS_PHONG = VS_LIT;

export const FS_PHONG = `#version 300 es
precision highp float;
in vec3 vWorld;
in vec3 vNrm;
out vec4 frag;
uniform vec3 uEye;
uniform vec3 uAlbedo;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uPointPos;
uniform vec3 uPointColor;
uniform float uShininess;
uniform float uAmbient;
uniform float uSpec;

vec3 shade(vec3 n, vec3 l, vec3 v, vec3 lightCol, float atten){
  float diff = max(dot(n, l), 0.0);
  vec3 h = normalize(l + v);
  float spec = pow(max(dot(n, h), 0.0), uShininess) * uSpec;
  return (uAlbedo * diff + spec * vec3(1.0)) * lightCol * atten;
}

void main(){
  vec3 n = normalize(vNrm);
  vec3 v = normalize(uEye - vWorld);
  vec3 col = uAlbedo * uAmbient;
  col += shade(n, normalize(-uLightDir), v, uLightColor, 1.0);
  vec3 toP = uPointPos - vWorld;
  float d = length(toP);
  col += shade(n, toP / max(d, 1e-4), v, uPointColor, 1.0 / (1.0 + 0.22*d + 0.04*d*d));
  frag = vec4(col, 1.0);
}`;

export const phongLab: LabDefinition = {
  id: "phong",
  index: "02",
  title: "Blinn-Phong",
  subtitle: "Normals · lights · BRDF",
  pipeline: ["VS", "Interpolation", "FS lighting", "Framebuffer"],
  supportsShaderEdit: true,
  params: [
    { key: "shininess", label: "Shininess", min: 4, max: 128, step: 1, default: 32 },
    { key: "spec", label: "Specular", min: 0, max: 2, step: 0.01, default: 0.7 },
    { key: "ambient", label: "Ambient", min: 0, max: 0.4, step: 0.01, default: 0.08 },
    { key: "spin", label: "Light orbit", min: 0, max: 2, step: 0.01, default: 0.5 },
  ],
  note: {
    title: "Per-pixel lighting on the fragment shader",
    body: "Blinn-Phong evaluates a Lambert diffuse term plus a specular lobe on the half-vector H = normalize(L+V). Doing this in the fragment shader (not the vertex shader) is what makes highlights round on a coarse mesh. Two lights: a directional (sun) and an attenuated point. The normal matrix is the inverse-transpose of the upper 3×3 of M, so non-uniform scale does not shear lighting.",
    glsl: "vec3 H = normalize(L + V);\nfloat spec = pow(max(dot(N, H), 0.0), shininess);",
    mapping: "gl_NormalMatrix → inverse-transpose of model\nfixed-function glLight is gone; you write the BRDF",
  },
  create(ctx: GlContext): LabInstance {
    const { gl } = ctx;
    const sphere = uploadMesh(gl, createIcosphere(0.85, 3));
    const torus = uploadMesh(gl, createTorus(0.7, 0.26, 48, 24));
    const cube = uploadMesh(gl, createCube(0.85));
    const plane = uploadMesh(gl, createPlane(14, 1, 0));
    const grid = makeGrid(gl);
    const lines = makeLineProgram(gl);
    let prog = createProgram(gl, VS_PHONG, FS_PHONG);
    let vs = VS_PHONG, fs = FS_PHONG;
    const model = mat4Identity();
    const normal = mat4Identity();
    const tmp = mat4Identity();
    const tmp2 = mat4Identity();
    let t = 0;
    let cam: Camera | null = null;
    let shininess = 32, spec = 0.7, ambient = 0.08, spin = 0.5;

    const drawOne = (mesh: typeof sphere, albedo: number[], tris: number) => {
      if (!cam) return 0;
      gl.useProgram(prog.prog);
      setCameraUniforms(gl, prog, cam, model);
      gl.uniformMatrix4fv(prog.uniforms.uNormal, false, mat4Normal(normal, model));
      gl.uniform3fv(prog.uniforms.uAlbedo, albedo);
      gl.uniform3fv(prog.uniforms.uEye, cam.eye);
      const lx = Math.cos(t * spin) * 0.6, lz = Math.sin(t * spin) * 0.6;
      gl.uniform3f(prog.uniforms.uLightDir, lx, -0.85, lz);
      gl.uniform3f(prog.uniforms.uLightColor, 1.0, 0.96, 0.88);
      gl.uniform3f(prog.uniforms.uPointPos, Math.sin(t * 0.7) * 2.4, 1.6, Math.cos(t * 0.7) * 2.4);
      gl.uniform3f(prog.uniforms.uPointColor, 0.45, 0.85, 0.7);
      gl.uniform1f(prog.uniforms.uShininess, shininess);
      gl.uniform1f(prog.uniforms.uAmbient, ambient);
      gl.uniform1f(prog.uniforms.uSpec, spec);
      drawMesh(gl, mesh);
      return tris;
    };

    return {
      getShader: () => ({ vs, fs }),
      setShader: (nvs, nfs) => {
        const r = tryCreateProgram(gl, nvs, nfs);
        if (!r.ok) return r.error;
        destroyProgram(gl, prog);
        prog = r.program;
        vs = nvs;
        fs = nfs;
        return null;
      },
      update(dt, _i, params, camera) {
        t += dt;
        cam = camera;
        shininess = params.shininess ?? 32;
        spec = params.spec ?? 0.7;
        ambient = params.ambient ?? 0.08;
        spin = params.spin ?? 0.5;
      },
      draw() {
        if (!cam) return { draws: 0, tris: 0, instances: 0, gpuMs: null, cpuMs: 0 };
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        drawGrid(gl, grid, lines, cam);
        let draws = 1, tris = 0;
        mat4FromTranslation(model, [0, 0.86, 0]);
        tris += drawOne(sphere, [0.82, 0.28, 0.26], sphere.count / 3);
        draws++;
        mat4FromTranslation(tmp, [-2.1, 0.7, 0.4]);
        mat4FromRotationY(tmp2, t * 0.4);
        mat4Multiply(model, tmp, tmp2);
        tris += drawOne(torus, [0.72, 0.74, 0.7], torus.count / 3);
        draws++;
        mat4FromTranslation(tmp, [2.15, 0.45, 0.2]);
        mat4FromRotationY(tmp2, t * 0.25);
        mat4Multiply(model, tmp, tmp2);
        tris += drawOne(cube, [0.35, 0.55, 0.72], 12);
        draws++;
        mat4Identity(model);
        tris += drawOne(plane, [0.18, 0.19, 0.2], 2);
        draws++;
        return { draws, tris, instances: 4, gpuMs: null, cpuMs: 0, extra: { lights: 2 } };
      },
      dispose() {
        disposeCommon(gl, [sphere, torus, cube, plane, grid], [prog, lines]);
      },
    };
  },
};
