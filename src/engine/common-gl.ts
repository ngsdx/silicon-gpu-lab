import { createMesh, createRawMesh, destroyMesh, drawMesh, type GpuMesh } from "./buffer";
import { createProgram, destroyProgram, type ShaderProgram } from "./shader";
import { createAxisLines, createGridLines, type CpuMesh } from "./mesh";
import type { Camera } from "./camera";

export const VS_LIT = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec2 aUv;
layout(location=3) in vec4 aTan;
uniform mat4 uModel, uView, uProj, uNormal;
out vec3 vWorld;
out vec3 vNrm;
out vec2 vUv;
out mat3 vTBN;
void main(){
  vec4 w = uModel * vec4(aPos,1.0);
  vWorld = w.xyz;
  vNrm = mat3(uNormal) * aNrm;
  vUv = aUv;
  vec3 T = normalize(mat3(uNormal) * aTan.xyz);
  vec3 N = normalize(vNrm);
  vec3 B = cross(N, T) * aTan.w;
  vTBN = mat3(T, B, N);
  gl_Position = uProj * uView * w;
}`;

export const FS_UNLIT = `#version 300 es
precision highp float;
in vec3 vNrm;
out vec4 frag;
uniform vec3 uColor;
void main(){
  vec3 n = normalize(vNrm);
  vec3 L = normalize(vec3(0.35, 0.82, 0.4));
  float ndl = max(dot(n, L), 0.0);
  float hemi = n.y * 0.5 + 0.5;
  frag = vec4(uColor * (0.18 + 0.35 * hemi + 0.55 * ndl), 1.0);
}`;

export const VS_LINE = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
uniform mat4 uView, uProj;
out vec3 vCol;
void main(){
  vCol = aNrm;
  gl_Position = uProj * uView * vec4(aPos,1.0);
}`;

export const FS_LINE = `#version 300 es
precision highp float;
in vec3 vCol;
out vec4 frag;
void main(){ frag = vec4(vCol, 1.0); }`;

export const VS_BLIT = `#version 300 es
layout(location=0) in vec2 aPos;
out vec2 vUv;
void main(){
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

export const FS_BLIT = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTex;
out vec4 frag;
void main(){ frag = texture(uTex, vUv); }`;

export function uploadMesh(gl: WebGL2RenderingContext, cpu: CpuMesh): GpuMesh {
  return createMesh(gl, cpu.vertices, cpu.indices, gl.TRIANGLES);
}

export function makeGrid(gl: WebGL2RenderingContext): GpuMesh {
  const g = createGridLines(10, 1);
  return createMesh(gl, g.vertices, undefined, gl.LINES);
}

export function makeAxes(gl: WebGL2RenderingContext): GpuMesh {
  const a = createAxisLines(1.8);
  return createMesh(gl, a.vertices, undefined, gl.LINES);
}

export function makeFullscreenTri(gl: WebGL2RenderingContext): GpuMesh {
  return createRawMesh(gl, new Float32Array([-1, -1, 3, -1, -1, 3]), 2, gl.TRIANGLES);
}

export function setCameraUniforms(gl: WebGL2RenderingContext, p: ShaderProgram, cam: Camera, model?: Float32Array) {
  if (p.uniforms.uView) gl.uniformMatrix4fv(p.uniforms.uView, false, cam.view);
  if (p.uniforms.uProj) gl.uniformMatrix4fv(p.uniforms.uProj, false, cam.proj);
  if (model && p.uniforms.uModel) gl.uniformMatrix4fv(p.uniforms.uModel, false, model);
}

export function drawGrid(gl: WebGL2RenderingContext, grid: GpuMesh, lineProg: ShaderProgram, cam: Camera) {
  gl.useProgram(lineProg.prog);
  setCameraUniforms(gl, lineProg, cam);
  gl.disable(gl.CULL_FACE);
  drawMesh(gl, grid);
  gl.enable(gl.CULL_FACE);
}

export function makeLineProgram(gl: WebGL2RenderingContext) {
  return createProgram(gl, VS_LINE, FS_LINE);
}

export function makeBlitProgram(gl: WebGL2RenderingContext) {
  return createProgram(gl, VS_BLIT, FS_BLIT);
}

export function blit(gl: WebGL2RenderingContext, blitProg: ShaderProgram, tri: GpuMesh, tex: WebGLTexture) {
  gl.disable(gl.DEPTH_TEST);
  gl.useProgram(blitProg.prog);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.uniform1i(blitProg.uniforms.uTex, 0);
  drawMesh(gl, tri);
  gl.enable(gl.DEPTH_TEST);
}

export function disposeCommon(
  gl: WebGL2RenderingContext,
  meshes: GpuMesh[],
  progs: (ShaderProgram | null)[],
) {
  for (const m of meshes) destroyMesh(gl, m);
  for (const p of progs) destroyProgram(gl, p);
}
