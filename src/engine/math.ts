/**
 * Column-major linear algebra matching the OpenGL / WebGL2 convention.
 * No third-party math library — every matrix you upload is built here.
 */

export type Vec3 = Float32Array;
export type Vec4 = Float32Array;
export type Mat4 = Float32Array;

export const DEG = Math.PI / 180;
export const RAD = 180 / Math.PI;

export function vec3(x = 0, y = 0, z = 0): Vec3 {
  return new Float32Array([x, y, z]);
}

export function vec4(x = 0, y = 0, z = 0, w = 1): Vec4 {
  return new Float32Array([x, y, z, w]);
}

export function vec3Set(out: Vec3, x: number, y: number, z: number): Vec3 {
  out[0] = x;
  out[1] = y;
  out[2] = z;
  return out;
}

export function vec3Copy(out: Vec3, a: ArrayLike<number>): Vec3 {
  out[0] = a[0]!;
  out[1] = a[1]!;
  out[2] = a[2]!;
  return out;
}

export function vec3Add(out: Vec3, a: ArrayLike<number>, b: ArrayLike<number>): Vec3 {
  out[0] = a[0]! + b[0]!;
  out[1] = a[1]! + b[1]!;
  out[2] = a[2]! + b[2]!;
  return out;
}

export function vec3Sub(out: Vec3, a: ArrayLike<number>, b: ArrayLike<number>): Vec3 {
  out[0] = a[0]! - b[0]!;
  out[1] = a[1]! - b[1]!;
  out[2] = a[2]! - b[2]!;
  return out;
}

export function vec3Scale(out: Vec3, a: ArrayLike<number>, s: number): Vec3 {
  out[0] = a[0]! * s;
  out[1] = a[1]! * s;
  out[2] = a[2]! * s;
  return out;
}

export function vec3Mad(out: Vec3, a: ArrayLike<number>, b: ArrayLike<number>, s: number): Vec3 {
  out[0] = a[0]! + b[0]! * s;
  out[1] = a[1]! + b[1]! * s;
  out[2] = a[2]! + b[2]! * s;
  return out;
}

export function vec3Dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
}

export function vec3Len(a: ArrayLike<number>): number {
  return Math.hypot(a[0]!, a[1]!, a[2]!);
}

export function vec3Normalize(out: Vec3, a: ArrayLike<number>): Vec3 {
  const l = Math.hypot(a[0]!, a[1]!, a[2]!) || 1;
  out[0] = a[0]! / l;
  out[1] = a[1]! / l;
  out[2] = a[2]! / l;
  return out;
}

export function vec3Cross(out: Vec3, a: ArrayLike<number>, b: ArrayLike<number>): Vec3 {
  const ax = a[0]!, ay = a[1]!, az = a[2]!;
  const bx = b[0]!, by = b[1]!, bz = b[2]!;
  out[0] = ay * bz - az * by;
  out[1] = az * bx - ax * bz;
  out[2] = ax * by - ay * bx;
  return out;
}

export function vec3Lerp(out: Vec3, a: ArrayLike<number>, b: ArrayLike<number>, t: number): Vec3 {
  out[0] = a[0]! + (b[0]! - a[0]!) * t;
  out[1] = a[1]! + (b[1]! - a[1]!) * t;
  out[2] = a[2]! + (b[2]! - a[2]!) * t;
  return out;
}

export function mat4Identity(out: Mat4 = new Float32Array(16)): Mat4 {
  out.fill(0);
  out[0] = out[5] = out[10] = out[15] = 1;
  return out;
}

export function mat4Copy(out: Mat4, a: Mat4): Mat4 {
  out.set(a);
  return out;
}

export function mat4Multiply(out: Mat4, a: ArrayLike<number>, b: ArrayLike<number>): Mat4 {
  const a00 = a[0]!, a01 = a[1]!, a02 = a[2]!, a03 = a[3]!;
  const a10 = a[4]!, a11 = a[5]!, a12 = a[6]!, a13 = a[7]!;
  const a20 = a[8]!, a21 = a[9]!, a22 = a[10]!, a23 = a[11]!;
  const a30 = a[12]!, a31 = a[13]!, a32 = a[14]!, a33 = a[15]!;

  const t = out === a || out === b ? new Float32Array(16) : out;

  for (let i = 0; i < 4; i++) {
    const bi0 = b[i * 4]!, bi1 = b[i * 4 + 1]!, bi2 = b[i * 4 + 2]!, bi3 = b[i * 4 + 3]!;
    t[i * 4] = a00 * bi0 + a10 * bi1 + a20 * bi2 + a30 * bi3;
    t[i * 4 + 1] = a01 * bi0 + a11 * bi1 + a21 * bi2 + a31 * bi3;
    t[i * 4 + 2] = a02 * bi0 + a12 * bi1 + a22 * bi2 + a32 * bi3;
    t[i * 4 + 3] = a03 * bi0 + a13 * bi1 + a23 * bi2 + a33 * bi3;
  }
  if (t !== out) out.set(t);
  return out;
}

export function mat4Translate(out: Mat4, a: Mat4, v: ArrayLike<number>): Mat4 {
  mat4Copy(out, a);
  out[12] = a[0]! * v[0]! + a[4]! * v[1]! + a[8]! * v[2]! + a[12]!;
  out[13] = a[1]! * v[0]! + a[5]! * v[1]! + a[9]! * v[2]! + a[13]!;
  out[14] = a[2]! * v[0]! + a[6]! * v[1]! + a[10]! * v[2]! + a[14]!;
  out[15] = a[3]! * v[0]! + a[7]! * v[1]! + a[11]! * v[2]! + a[15]!;
  return out;
}

export function mat4FromTranslation(out: Mat4, v: ArrayLike<number>): Mat4 {
  mat4Identity(out);
  out[12] = v[0]!;
  out[13] = v[1]!;
  out[14] = v[2]!;
  return out;
}

export function mat4FromRotationY(out: Mat4, r: number): Mat4 {
  const c = Math.cos(r), s = Math.sin(r);
  mat4Identity(out);
  out[0] = c;
  out[2] = -s;
  out[8] = s;
  out[10] = c;
  return out;
}

export function mat4FromRotationX(out: Mat4, r: number): Mat4 {
  const c = Math.cos(r), s = Math.sin(r);
  mat4Identity(out);
  out[5] = c;
  out[6] = s;
  out[9] = -s;
  out[10] = c;
  return out;
}

export function mat4FromRotationZ(out: Mat4, r: number): Mat4 {
  const c = Math.cos(r), s = Math.sin(r);
  mat4Identity(out);
  out[0] = c;
  out[1] = s;
  out[4] = -s;
  out[5] = c;
  return out;
}

export function mat4FromScaling(out: Mat4, v: ArrayLike<number>): Mat4 {
  mat4Identity(out);
  out[0] = v[0]!;
  out[5] = v[1]!;
  out[10] = v[2]!;
  return out;
}

export function mat4TRS(
  out: Mat4,
  t: ArrayLike<number>,
  eulerYXZ: ArrayLike<number>,
  s: ArrayLike<number>,
): Mat4 {
  const rx = mat4FromRotationX(new Float32Array(16), eulerYXZ[0]!);
  const ry = mat4FromRotationY(new Float32Array(16), eulerYXZ[1]!);
  const rz = mat4FromRotationZ(new Float32Array(16), eulerYXZ[2]!);
  const sc = mat4FromScaling(new Float32Array(16), s);
  const rot = mat4Multiply(new Float32Array(16), ry, mat4Multiply(new Float32Array(16), rx, rz));
  mat4Multiply(out, rot, sc);
  out[12] = t[0]!;
  out[13] = t[1]!;
  out[14] = t[2]!;
  return out;
}

/** Symmetric perspective. fovy in radians. Maps to clip-space WebGL (z in [-w, w] NDC). */
export function mat4Perspective(out: Mat4, fovy: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) * nf;
  out[11] = -1;
  out[14] = 2 * far * near * nf;
  return out;
}

export function mat4Ortho(
  out: Mat4,
  left: number,
  right: number,
  bottom: number,
  top: number,
  near: number,
  far: number,
): Mat4 {
  out.fill(0);
  out[0] = 2 / (right - left);
  out[5] = 2 / (top - bottom);
  out[10] = -2 / (far - near);
  out[12] = -(right + left) / (right - left);
  out[13] = -(top + bottom) / (top - bottom);
  out[14] = -(far + near) / (far - near);
  out[15] = 1;
  return out;
}

/**
 * View matrix. Camera looks along -Z of its local frame (OpenGL convention).
 * Yaw 0 faces world -Z; +yaw is CCW about +Y.
 */
export function mat4LookAt(out: Mat4, eye: ArrayLike<number>, target: ArrayLike<number>, up: ArrayLike<number>): Mat4 {
  const zx = eye[0]! - target[0]!;
  const zy = eye[1]! - target[1]!;
  const zz = eye[2]! - target[2]!;
  let zl = Math.hypot(zx, zy, zz) || 1;
  const z0 = zx / zl, z1 = zy / zl, z2 = zz / zl;

  let x0 = up[1]! * z2 - up[2]! * z1;
  let x1 = up[2]! * z0 - up[0]! * z2;
  let x2 = up[0]! * z1 - up[1]! * z0;
  let xl = Math.hypot(x0, x1, x2) || 1;
  x0 /= xl;
  x1 /= xl;
  x2 /= xl;

  const y0 = z1 * x2 - z2 * x1;
  const y1 = z2 * x0 - z0 * x2;
  const y2 = z0 * x1 - z1 * x0;

  out[0] = x0;
  out[1] = y0;
  out[2] = z0;
  out[3] = 0;
  out[4] = x1;
  out[5] = y1;
  out[6] = z1;
  out[7] = 0;
  out[8] = x2;
  out[9] = y2;
  out[10] = z2;
  out[11] = 0;
  out[12] = -(x0 * eye[0]! + x1 * eye[1]! + x2 * eye[2]!);
  out[13] = -(y0 * eye[0]! + y1 * eye[1]! + y2 * eye[2]!);
  out[14] = -(z0 * eye[0]! + z1 * eye[1]! + z2 * eye[2]!);
  out[15] = 1;
  return out;
}

export function mat4Invert(out: Mat4, a: ArrayLike<number>): Mat4 | null {
  const a00 = a[0]!, a01 = a[1]!, a02 = a[2]!, a03 = a[3]!;
  const a10 = a[4]!, a11 = a[5]!, a12 = a[6]!, a13 = a[7]!;
  const a20 = a[8]!, a21 = a[9]!, a22 = a[10]!, a23 = a[11]!;
  const a30 = a[12]!, a31 = a[13]!, a32 = a[14]!, a33 = a[15]!;

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return null;
  det = 1 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

export function mat4Transpose(out: Mat4, a: ArrayLike<number>): Mat4 {
  const t = out === a ? new Float32Array(a) : a;
  out[0] = t[0]!;
  out[1] = t[4]!;
  out[2] = t[8]!;
  out[3] = t[12]!;
  out[4] = t[1]!;
  out[5] = t[5]!;
  out[6] = t[9]!;
  out[7] = t[13]!;
  out[8] = t[2]!;
  out[9] = t[6]!;
  out[10] = t[10]!;
  out[11] = t[14]!;
  out[12] = t[3]!;
  out[13] = t[7]!;
  out[14] = t[11]!;
  out[15] = t[15]!;
  return out;
}

/** Inverse-transpose of the upper 3x3, packed as mat4 (last row/col identity). */
export function mat4Normal(out: Mat4, model: ArrayLike<number>): Mat4 {
  const inv = mat4Invert(new Float32Array(16), model);
  if (!inv) return mat4Identity(out);
  return mat4Transpose(out, inv);
}

export function mat4TransformVec3(out: Vec3, m: ArrayLike<number>, v: ArrayLike<number>, w = 1): Vec3 {
  const x = v[0]!, y = v[1]!, z = v[2]!;
  const rw = m[3]! * x + m[7]! * y + m[11]! * z + m[15]! * w;
  out[0] = m[0]! * x + m[4]! * y + m[8]! * z + m[12]! * w;
  out[1] = m[1]! * x + m[5]! * y + m[9]! * z + m[13]! * w;
  out[2] = m[2]! * x + m[6]! * y + m[10]! * z + m[14]! * w;
  if (w !== 0 && rw !== 0 && rw !== 1) {
    out[0] /= rw;
    out[1] /= rw;
    out[2] /= rw;
  }
  return out;
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function wrapAngle(a: number): number {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

export function hash01(n: number): number {
  const x = Math.sin(n * 127.1) * 43758.5453;
  return x - Math.floor(x);
}
