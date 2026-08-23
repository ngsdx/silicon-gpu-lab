import { destroyMesh, drawMesh, type GpuMesh } from "@/engine/buffer";
import { createProgram, destroyProgram, tryCreateProgram, type ShaderProgram } from "@/engine/shader";
import type { LabDefinition, LabInstance } from "@/engine/lab-types";
import type { GlContext } from "@/engine/gpu";

export const VS_DEFAULT = `#version 300 es
layout(location=0) in vec2 aPos;
layout(location=1) in vec3 aCol;
uniform float uScale;
out vec3 vCol;
void main(){
  vCol = aCol;
  gl_Position = vec4(aPos * uScale, 0.0, 1.0);
}`;

export const FS_DEFAULT = `#version 300 es
precision highp float;
in vec3 vCol;
out vec4 frag;
uniform float uTime;
uniform float uPulse;
void main(){
  float g = 0.5 + 0.5 * sin(uTime * 2.0);
  vec3 c = mix(vCol, vCol * (0.7 + 0.3 * g), uPulse);
  frag = vec4(c, 1.0);
}`;

export const helloTriangleLab: LabDefinition = {
  id: "triangle",
  index: "00",
  title: "Hello Triangle",
  subtitle: "VAO · VBO · GLSL ES 3.00",
  pipeline: ["Vertex fetch", "Vertex shader", "Rasterizer", "Fragment shader", "ROP"],
  supportsShaderEdit: true,
  params: [
    { key: "pulse", label: "Pulse mix", min: 0, max: 1, step: 0.01, default: 0.35 },
    { key: "scale", label: "NDC scale", min: 0.35, max: 1.35, step: 0.01, default: 1 },
  ],
  note: {
    title: "The first draw call",
    body: "Every frame of every game starts here: three vertices in a buffer, a compiled shader pair, a draw. WebGL2 is OpenGL ES 3.00 — the same pipeline, objects, and GLSL you would drive from C with GLFW. The GPU fetches bytes, runs the vertex stage in parallel, interpolates varyings across the triangle, then shades every fragment. Edit the GLSL on the right and hit Apply — a failed compile keeps the last good program, the same way a driver would.",
    glsl: "layout(location=0) in vec2 aPos;\nlayout(location=1) in vec3 aCol;\nout vec3 vCol; // interpolated by the rasterizer",
    mapping: "glGenVertexArrays  → createVertexArray()\nglBindBuffer(GL_ARRAY_BUFFER) → bindBuffer(ARRAY_BUFFER)\nglDrawArrays(GL_TRIANGLES,0,3) → drawArrays(TRIANGLES,0,3)",
  },
  create(ctx: GlContext): LabInstance {
    const { gl } = ctx;
    const verts = new Float32Array([
      0.0, 0.64, 0.93, 0.27, 0.27,
      -0.7, -0.54, 0.27, 0.84, 0.4,
      0.7, -0.54, 0.3, 0.42, 0.93,
    ]);
    const vao = gl.createVertexArray()!;
    const vbo = gl.createBuffer()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 20, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 20, 8);
    gl.bindVertexArray(null);
    const mesh: GpuMesh = { vao, vbo, ibo: null, count: 3, mode: gl.TRIANGLES, stride: 20 };

    let prog: ShaderProgram = createProgram(gl, VS_DEFAULT, FS_DEFAULT);
    let vs = VS_DEFAULT;
    let fs = FS_DEFAULT;
    let t = 0;
    let pulse = 0.35;
    let scale = 1;

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
      update(dt, _i, params) {
        t += dt;
        pulse = params.pulse ?? 0.35;
        scale = params.scale ?? 1;
      },
      draw() {
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.CULL_FACE);
        gl.clearColor(0.035, 0.039, 0.043, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(prog.prog);
        if (prog.uniforms.uTime) gl.uniform1f(prog.uniforms.uTime, t);
        if (prog.uniforms.uPulse) gl.uniform1f(prog.uniforms.uPulse, pulse);
        if (prog.uniforms.uScale) gl.uniform1f(prog.uniforms.uScale, scale);
        drawMesh(gl, mesh);
        gl.enable(gl.DEPTH_TEST);
        gl.enable(gl.CULL_FACE);
        return { draws: 1, tris: 1, instances: 1, gpuMs: null, cpuMs: 0 };
      },
      dispose() {
        destroyMesh(gl, mesh);
        destroyProgram(gl, prog);
      },
    };
  },
};
