export type ShaderProgram = {
  prog: WebGLProgram;
  uniforms: Record<string, WebGLUniformLocation | null>;
  attribs: Record<string, number>;
};

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type);
  if (!sh) throw new Error("createShader failed");
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh) ?? "compile error";
    gl.deleteShader(sh);
    throw new Error(log);
  }
  return sh;
}

export function createProgram(
  gl: WebGL2RenderingContext,
  vsSrc: string,
  fsSrc: string,
  attribBindings?: Record<string, number>,
): ShaderProgram {
  const vs = compile(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fsSrc);
  const prog = gl.createProgram();
  if (!prog) throw new Error("createProgram failed");
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  if (attribBindings) {
    for (const [name, loc] of Object.entries(attribBindings)) {
      gl.bindAttribLocation(prog, loc, name);
    }
  }
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog) ?? "link error";
    gl.deleteProgram(prog);
    throw new Error(log);
  }

  const uniforms: Record<string, WebGLUniformLocation | null> = {};
  const nU = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS) as number;
  for (let i = 0; i < nU; i++) {
    const info = gl.getActiveUniform(prog, i);
    if (!info) continue;
    const name = info.name.replace(/\[0]$/, "");
    uniforms[name] = gl.getUniformLocation(prog, info.name);
  }

  const attribs: Record<string, number> = {};
  const nA = gl.getProgramParameter(prog, gl.ACTIVE_ATTRIBUTES) as number;
  for (let i = 0; i < nA; i++) {
    const info = gl.getActiveAttrib(prog, i);
    if (!info) continue;
    attribs[info.name] = gl.getAttribLocation(prog, info.name);
  }

  return { prog, uniforms, attribs };
}

export function tryCreateProgram(
  gl: WebGL2RenderingContext,
  vsSrc: string,
  fsSrc: string,
): { ok: true; program: ShaderProgram } | { ok: false; error: string } {
  try {
    return { ok: true, program: createProgram(gl, vsSrc, fsSrc) };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}

export function destroyProgram(gl: WebGL2RenderingContext, p: ShaderProgram | null): void {
  if (p) gl.deleteProgram(p.prog);
}

export const GLSL = {
  vs: (body: string) => `#version 300 es\n${body}`,
  fs: (body: string) => `#version 300 es\nprecision highp float;\n${body}`,
};
