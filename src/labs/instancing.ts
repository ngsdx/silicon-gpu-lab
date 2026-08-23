import { drawMeshInstanced } from "@/engine/buffer";
import { createProgram } from "@/engine/shader";
import { disposeCommon, drawGrid, makeGrid, makeLineProgram, uploadMesh } from "@/engine/common-gl";
import { createIcosphere } from "@/engine/mesh";
import { hash01, mat4TRS } from "@/engine/math";
import type { Camera } from "@/engine/camera";
import type { LabDefinition, LabInstance } from "@/engine/lab-types";
import type { GlContext } from "@/engine/gpu";

const VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=4) in vec4 iM0;
layout(location=5) in vec4 iM1;
layout(location=6) in vec4 iM2;
layout(location=7) in vec4 iM3;
layout(location=8) in vec3 iCol;
uniform mat4 uView, uProj;
out vec3 vNrm; out vec3 vCol; out vec3 vWorld;
void main(){
  mat4 iM = mat4(iM0, iM1, iM2, iM3);
  vec4 w = iM * vec4(aPos,1.0);
  vWorld = w.xyz;
  vNrm = mat3(iM) * aNrm;
  vCol = iCol;
  gl_Position = uProj * uView * w;
}`;

const FS = `#version 300 es
precision highp float;
in vec3 vNrm; in vec3 vCol; in vec3 vWorld;
uniform vec3 uEye;
out vec4 frag;
void main(){
  vec3 n = normalize(vNrm);
  vec3 L = normalize(vec3(0.4, 0.85, 0.25));
  float diff = max(dot(n, L), 0.0);
  float hemi = n.y * 0.5 + 0.5;
  float spec = pow(max(dot(n, normalize(L + normalize(uEye - vWorld))), 0.0), 32.0) * 0.25;
  frag = vec4(vCol * (0.12 + 0.55 * hemi + 0.45 * diff) + spec, 1.0);
}`;

const MAX = 4096;

export const instancingLab: LabDefinition = {
  id: "instancing",
  index: "05",
  title: "Instancing",
  subtitle: "One mesh · N matrices · 1 draw",
  pipeline: ["Instance attribs", "VS × N", "Raster", "FS"],
  params: [
    { key: "count", label: "Instances", min: 64, max: MAX, step: 64, default: 2048 },
    { key: "spread", label: "Spread", min: 4, max: 22, step: 0.5, default: 12 },
    { key: "spin", label: "Spin", min: 0, max: 2, step: 0.01, default: 0.35 },
  ],
  note: {
    title: "Amortize the draw call",
    body: "A draw call is a CPU→driver→command buffer handshake. Instancing uploads a per-instance matrix (four vec4 attributes, divisor 1) and issues one glDrawElementsInstanced. 2048 meshes, one VAO, one program bind. The vertex shader sees gl_InstanceID implicitly via those attributes. This is how a GPU actually draws forests, debris, particles, and GPU-driven grass.",
    glsl: "layout(location=4) in vec4 iM0; // column 0 of instance M\nlayout(location=8) in vec3 iCol;\ngl_Position = uProj * uView * iM * vec4(aPos,1);",
    mapping: "glVertexAttribDivisor(4, 1)\nglDrawElementsInstanced(GL_TRIANGLES, count, GL_UNSIGNED_INT, 0, N)",
  },
  create(ctx: GlContext): LabInstance {
    const { gl } = ctx;
    const cpu = createIcosphere(0.22, 1);
    const mesh = uploadMesh(gl, cpu);
    const grid = makeGrid(gl);
    const lines = makeLineProgram(gl);
    const prog = createProgram(gl, VS, FS);

    const matData = new Float32Array(MAX * 16);
    const colData = new Float32Array(MAX * 3);
    const seeds: { y: number; s: number; r: number; g: number; b: number; yaw: number; x: number; z: number }[] = [];
    for (let i = 0; i < MAX; i++) {
      const a = hash01(i * 3.1) * Math.PI * 2;
      const rad = Math.sqrt(hash01(i * 7.7)) * 10;
      seeds.push({
        x: Math.cos(a) * rad,
        z: Math.sin(a) * rad,
        y: hash01(i * 11.3) * 0.4,
        s: 0.45 + hash01(i * 4.4) * 1.4,
        r: 0.35 + hash01(i * 1.1) * 0.5,
        g: 0.4 + hash01(i * 2.2) * 0.45,
        b: 0.38 + hash01(i * 3.3) * 0.4,
        yaw: hash01(i * 9.9) * Math.PI * 2,
      });
      colData[i * 3] = seeds[i]!.r;
      colData[i * 3 + 1] = seeds[i]!.g;
      colData[i * 3 + 2] = seeds[i]!.b;
    }

    const matBuf = gl.createBuffer()!;
    const colBuf = gl.createBuffer()!;
    gl.bindVertexArray(mesh.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, matBuf);
    gl.bufferData(gl.ARRAY_BUFFER, matData, gl.DYNAMIC_DRAW);
    for (let c = 0; c < 4; c++) {
      const loc = 4 + c;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 64, c * 16);
      gl.vertexAttribDivisor(loc, 1);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
    gl.bufferData(gl.ARRAY_BUFFER, colData, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(8);
    gl.vertexAttribPointer(8, 3, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(8, 1);
    gl.bindVertexArray(null);

    const tmp = new Float32Array(16);
    let cam: Camera | null = null;
    let count = 2048, spread = 12, spin = 0.35, t = 0;

    return {
      update(dt, _i, params, camera) {
        t += dt;
        cam = camera;
        count = Math.max(1, Math.min(MAX, params.count ?? 2048)) | 0;
        spread = params.spread ?? 12;
        spin = params.spin ?? 0.35;
      },
      draw() {
        if (!cam) return { draws: 0, tris: 0, instances: 0, gpuMs: null, cpuMs: 0 };
        const t0 = performance.now();
        for (let i = 0; i < count; i++) {
          const s = seeds[i]!;
          const scale = s.s * (spread / 12);
          mat4TRS(tmp, [s.x * (spread / 10), 0.25 + s.y, s.z * (spread / 10)], [0, s.yaw + t * spin, 0], [scale, scale, scale]);
          matData.set(tmp, i * 16);
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, matBuf);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, matData.subarray(0, count * 16));

        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        drawGrid(gl, grid, lines, cam);
        gl.useProgram(prog.prog);
        gl.uniformMatrix4fv(prog.uniforms.uView, false, cam.view);
        gl.uniformMatrix4fv(prog.uniforms.uProj, false, cam.proj);
        gl.uniform3fv(prog.uniforms.uEye, cam.eye);
        drawMeshInstanced(gl, mesh, count);
        const cpuMs = performance.now() - t0;
        return {
          draws: 2,
          tris: (mesh.count / 3) * count,
          instances: count,
          gpuMs: null,
          cpuMs,
          extra: { "draw calls": 1, "verts/instance": mesh.count },
        };
      },
      dispose() {
        gl.deleteBuffer(matBuf);
        gl.deleteBuffer(colBuf);
        disposeCommon(gl, [mesh, grid], [prog, lines]);
      },
    };
  },
};
