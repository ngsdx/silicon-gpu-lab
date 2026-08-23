export type GpuMesh = {
  vao: WebGLVertexArrayObject;
  vbo: WebGLBuffer;
  ibo: WebGLBuffer | null;
  count: number;
  mode: number;
  stride: number;
};

/**
 * Interleaved vertex: p3 n3 uv2 t4  (12 floats, 48 bytes)
 * Matches desktop packed vertex layouts.
 */
export const VERTEX_STRIDE = 12;
export const VERTEX_BYTES = VERTEX_STRIDE * 4;

export function createMesh(
  gl: WebGL2RenderingContext,
  vertices: Float32Array,
  indices?: Uint16Array | Uint32Array,
  mode = 4 /* TRIANGLES */,
): GpuMesh {
  const vao = gl.createVertexArray();
  const vbo = gl.createBuffer();
  if (!vao || !vbo) throw new Error("buffer alloc failed");

  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

  const stride = VERTEX_BYTES;
  // loc 0 position
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
  // loc 1 normal
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 12);
  // loc 2 uv
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 2, gl.FLOAT, false, stride, 24);
  // loc 3 tangent (xyz + handedness)
  gl.enableVertexAttribArray(3);
  gl.vertexAttribPointer(3, 4, gl.FLOAT, false, stride, 32);

  let ibo: WebGLBuffer | null = null;
  let count: number;
  if (indices) {
    ibo = gl.createBuffer();
    if (!ibo) throw new Error("ibo alloc failed");
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    count = indices.length;
  } else {
    count = vertices.length / VERTEX_STRIDE;
  }

  gl.bindVertexArray(null);
  return { vao, vbo, ibo, count, mode, stride };
}

export function createRawMesh(
  gl: WebGL2RenderingContext,
  vertices: Float32Array,
  components: number,
  mode = 4,
): GpuMesh {
  const vao = gl.createVertexArray();
  const vbo = gl.createBuffer();
  if (!vao || !vbo) throw new Error("buffer alloc failed");
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, components, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return {
    vao,
    vbo,
    ibo: null,
    count: vertices.length / components,
    mode,
    stride: components * 4,
  };
}

export function drawMesh(gl: WebGL2RenderingContext, mesh: GpuMesh): void {
  gl.bindVertexArray(mesh.vao);
  if (mesh.ibo) {
    const type = gl.UNSIGNED_INT;
    gl.drawElements(mesh.mode, mesh.count, type, 0);
  } else {
    gl.drawArrays(mesh.mode, 0, mesh.count);
  }
}

export function drawMeshInstanced(gl: WebGL2RenderingContext, mesh: GpuMesh, instances: number): void {
  gl.bindVertexArray(mesh.vao);
  if (mesh.ibo) {
    gl.drawElementsInstanced(mesh.mode, mesh.count, gl.UNSIGNED_INT, 0, instances);
  } else {
    gl.drawArraysInstanced(mesh.mode, 0, mesh.count, instances);
  }
}

export function destroyMesh(gl: WebGL2RenderingContext, mesh: GpuMesh): void {
  gl.deleteVertexArray(mesh.vao);
  gl.deleteBuffer(mesh.vbo);
  if (mesh.ibo) gl.deleteBuffer(mesh.ibo);
}

export function createUbo(gl: WebGL2RenderingContext, bytes: number, binding = 0): WebGLBuffer {
  const ubo = gl.createBuffer();
  if (!ubo) throw new Error("ubo alloc failed");
  gl.bindBuffer(gl.UNIFORM_BUFFER, ubo);
  gl.bufferData(gl.UNIFORM_BUFFER, bytes, gl.DYNAMIC_DRAW);
  gl.bindBufferBase(gl.UNIFORM_BUFFER, binding, ubo);
  return ubo;
}
