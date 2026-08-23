import { VERTEX_STRIDE } from "./buffer";

export type CpuMesh = {
  vertices: Float32Array;
  indices: Uint32Array;
  triCount: number;
};

function pushVert(
  verts: number[],
  p: [number, number, number],
  n: [number, number, number],
  uv: [number, number],
  t: [number, number, number, number] = [1, 0, 0, 1],
) {
  verts.push(p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1], t[0], t[1], t[2], t[3]);
}

function pack(verts: number[], indices: number[]): CpuMesh {
  computeTangents(verts, indices);
  return {
    vertices: new Float32Array(verts),
    indices: new Uint32Array(indices),
    triCount: indices.length / 3,
  };
}

function computeTangents(verts: number[], indices: number[]): void {
  const tan1 = new Float32Array((verts.length / VERTEX_STRIDE) * 3);
  const tan2 = new Float32Array(tan1.length);
  for (let i = 0; i < indices.length; i += 3) {
    const i0 = indices[i]!, i1 = indices[i + 1]!, i2 = indices[i + 2]!;
    const o0 = i0 * VERTEX_STRIDE, o1 = i1 * VERTEX_STRIDE, o2 = i2 * VERTEX_STRIDE;
    const x0 = verts[o0]!, y0 = verts[o0 + 1]!, z0 = verts[o0 + 2]!;
    const x1 = verts[o1]!, y1 = verts[o1 + 1]!, z1 = verts[o1 + 2]!;
    const x2 = verts[o2]!, y2 = verts[o2 + 1]!, z2 = verts[o2 + 2]!;
    const u0 = verts[o0 + 6]!, v0 = verts[o0 + 7]!;
    const u1 = verts[o1 + 6]!, v1 = verts[o1 + 7]!;
    const u2 = verts[o2 + 6]!, v2 = verts[o2 + 7]!;
    const dx1 = x1 - x0, dy1 = y1 - y0, dz1 = z1 - z0;
    const dx2 = x2 - x0, dy2 = y2 - y0, dz2 = z2 - z0;
    const du1 = u1 - u0, dv1 = v1 - v0;
    const du2 = u2 - u0, dv2 = v2 - v0;
    const r = du1 * dv2 - du2 * dv1;
    const f = Math.abs(r) < 1e-8 ? 1 : 1 / r;
    const tx = (dx1 * dv2 - dx2 * dv1) * f;
    const ty = (dy1 * dv2 - dy2 * dv1) * f;
    const tz = (dz1 * dv2 - dz2 * dv1) * f;
    const bx = (dx2 * du1 - dx1 * du2) * f;
    const by = (dy2 * du1 - dy1 * du2) * f;
    const bz = (dz2 * du1 - dz1 * du2) * f;
    for (const idx of [i0, i1, i2]) {
      tan1[idx * 3] = (tan1[idx * 3] ?? 0) + tx;
      tan1[idx * 3 + 1] = (tan1[idx * 3 + 1] ?? 0) + ty;
      tan1[idx * 3 + 2] = (tan1[idx * 3 + 2] ?? 0) + tz;
      tan2[idx * 3] = (tan2[idx * 3] ?? 0) + bx;
      tan2[idx * 3 + 1] = (tan2[idx * 3 + 1] ?? 0) + by;
      tan2[idx * 3 + 2] = (tan2[idx * 3 + 2] ?? 0) + bz;
    }
  }
  const nVert = verts.length / VERTEX_STRIDE;
  for (let i = 0; i < nVert; i++) {
    const o = i * VERTEX_STRIDE;
    const nx = verts[o + 3]!, ny = verts[o + 4]!, nz = verts[o + 5]!;
    let tx = tan1[i * 3] ?? 1, ty = tan1[i * 3 + 1] ?? 0, tz = tan1[i * 3 + 2] ?? 0;
    const ndott = nx * tx + ny * ty + nz * tz;
    tx -= nx * ndott;
    ty -= ny * ndott;
    tz -= nz * ndott;
    const tl = Math.hypot(tx, ty, tz) || 1;
    tx /= tl;
    ty /= tl;
    tz /= tl;
    const bx = tan2[i * 3] ?? 0, by = tan2[i * 3 + 1] ?? 1, bz = tan2[i * 3 + 2] ?? 0;
    const cx = ny * tz - nz * ty;
    const cy = nz * tx - nx * tz;
    const cz = nx * ty - ny * tx;
    const w = cx * bx + cy * by + cz * bz < 0 ? -1 : 1;
    verts[o + 8] = tx;
    verts[o + 9] = ty;
    verts[o + 10] = tz;
    verts[o + 11] = w;
  }
}

export function createCube(s = 1): CpuMesh {
  const h = s / 2;
  const faces: { n: [number, number, number]; t: [number, number, number][]; uv: [number, number][] }[] = [
    { n: [0, 0, 1], t: [[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]], uv: [[0, 0], [1, 0], [1, 1], [0, 1]] },
    { n: [0, 0, -1], t: [[h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h]], uv: [[0, 0], [1, 0], [1, 1], [0, 1]] },
    { n: [0, 1, 0], t: [[-h, h, h], [h, h, h], [h, h, -h], [-h, h, -h]], uv: [[0, 0], [1, 0], [1, 1], [0, 1]] },
    { n: [0, -1, 0], t: [[-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h]], uv: [[0, 0], [1, 0], [1, 1], [0, 1]] },
    { n: [1, 0, 0], t: [[h, -h, h], [h, -h, -h], [h, h, -h], [h, h, h]], uv: [[0, 0], [1, 0], [1, 1], [0, 1]] },
    { n: [-1, 0, 0], t: [[-h, -h, -h], [-h, -h, h], [-h, h, h], [-h, h, -h]], uv: [[0, 0], [1, 0], [1, 1], [0, 1]] },
  ];
  const verts: number[] = [];
  const idx: number[] = [];
  let base = 0;
  for (const f of faces) {
    for (let i = 0; i < 4; i++) pushVert(verts, f.t[i]!, f.n, f.uv[i]!);
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
    base += 4;
  }
  return pack(verts, idx);
}

export function createPlane(size = 10, segs = 1, y = 0): CpuMesh {
  const verts: number[] = [];
  const idx: number[] = [];
  for (let z = 0; z <= segs; z++) {
    for (let x = 0; x <= segs; x++) {
      const u = x / segs, v = z / segs;
      pushVert(verts, [(u - 0.5) * size, y, (v - 0.5) * size], [0, 1, 0], [u, v]);
    }
  }
  const stride = segs + 1;
  for (let z = 0; z < segs; z++) {
    for (let x = 0; x < segs; x++) {
      const a = z * stride + x;
      idx.push(a, a + stride, a + stride + 1, a, a + stride + 1, a + 1);
    }
  }
  return pack(verts, idx);
}

export function createUvSphere(r = 1, stacks = 24, slices = 32): CpuMesh {
  const verts: number[] = [];
  const idx: number[] = [];
  for (let i = 0; i <= stacks; i++) {
    const phi = (i / stacks) * Math.PI;
    const sp = Math.sin(phi), cp = Math.cos(phi);
    for (let j = 0; j <= slices; j++) {
      const th = (j / slices) * Math.PI * 2;
      const st = Math.sin(th), ct = Math.cos(th);
      const x = ct * sp, y = cp, z = st * sp;
      pushVert(verts, [x * r, y * r, z * r], [x, y, z], [j / slices, 1 - i / stacks]);
    }
  }
  for (let i = 0; i < stacks; i++) {
    for (let j = 0; j < slices; j++) {
      const a = i * (slices + 1) + j;
      const b = a + slices + 1;
      idx.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  return pack(verts, idx);
}

export function createTorus(R = 1.1, r = 0.38, segs = 48, sides = 24): CpuMesh {
  const verts: number[] = [];
  const idx: number[] = [];
  for (let i = 0; i <= segs; i++) {
    const u = (i / segs) * Math.PI * 2;
    const cu = Math.cos(u), su = Math.sin(u);
    for (let j = 0; j <= sides; j++) {
      const v = (j / sides) * Math.PI * 2;
      const cv = Math.cos(v), sv = Math.sin(v);
      const x = (R + r * cv) * cu;
      const y = r * sv;
      const z = (R + r * cv) * su;
      const nx = cv * cu, ny = sv, nz = cv * su;
      pushVert(verts, [x, y, z], [nx, ny, nz], [i / segs, j / sides]);
    }
  }
  const stride = sides + 1;
  for (let i = 0; i < segs; i++) {
    for (let j = 0; j < sides; j++) {
      const a = i * stride + j;
      const b = a + stride;
      idx.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  return pack(verts, idx);
}

export function createIcosphere(r = 1, subdiv = 2): CpuMesh {
  const t = (1 + Math.sqrt(5)) / 2;
  const raw: [number, number, number][] = [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
    [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ];
  const norm = (p: [number, number, number]): [number, number, number] => {
    const l = Math.hypot(p[0], p[1], p[2]) || 1;
    return [p[0] / l, p[1] / l, p[2] / l];
  };
  let vertsPts = raw.map(norm);
  let faces: [number, number, number][] = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];
  const midCache = new Map<string, number>();
  const midpoint = (a: number, b: number) => {
    const key = a < b ? `${a}-${b}` : `${b}-${a}`;
    const hit = midCache.get(key);
    if (hit !== undefined) return hit;
    const pa = vertsPts[a]!, pb = vertsPts[b]!;
    const m = norm([(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2]);
    const idx = vertsPts.length;
    vertsPts.push(m);
    midCache.set(key, idx);
    return idx;
  };
  for (let s = 0; s < subdiv; s++) {
    const next: [number, number, number][] = [];
    for (const [a, b, c] of faces) {
      const ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
      next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
    }
    faces = next;
    midCache.clear();
  }
  const verts: number[] = [];
  const idx: number[] = [];
  vertsPts.forEach((p) => {
    const u = 0.5 + Math.atan2(p[2], p[0]) / (Math.PI * 2);
    const v = 0.5 - Math.asin(p[1]) / Math.PI;
    pushVert(verts, [p[0] * r, p[1] * r, p[2] * r], p, [u, v]);
  });
  for (const f of faces) idx.push(f[0], f[1], f[2]);
  return pack(verts, idx);
}

export function createGridLines(half = 8, step = 1): { vertices: Float32Array; count: number } {
  const lines: number[] = [];
  for (let i = -half; i <= half; i += step) {
    const major = i === 0;
    const c = major ? 0.42 : 0.16;
    lines.push(-half, 0, i, c, c, c + (major ? 0.08 : 0), 0, 0, 1, 0, 0, 1);
    lines.push(half, 0, i, c, c, c + (major ? 0.08 : 0), 0, 0, 1, 0, 0, 1);
    lines.push(i, 0, -half, c, c, c + (major ? 0.08 : 0), 0, 0, 1, 0, 0, 1);
    lines.push(i, 0, half, c, c, c + (major ? 0.08 : 0), 0, 0, 1, 0, 0, 1);
  }
  return { vertices: new Float32Array(lines), count: lines.length / VERTEX_STRIDE };
}

export function createAxisLines(len = 2): { vertices: Float32Array; count: number } {
  const l: number[] = [];
  const add = (a: number[], b: number[], c: number[]) => {
    pushVert(l, [a[0]!, a[1]!, a[2]!], [c[0]!, c[1]!, c[2]!], [0, 0]);
    pushVert(l, [b[0]!, b[1]!, b[2]!], [c[0]!, c[1]!, c[2]!], [1, 0]);
  };
  add([0, 0, 0], [len, 0, 0], [1, 0.25, 0.22]);
  add([0, 0, 0], [0, len, 0], [0.35, 0.85, 0.5]);
  add([0, 0, 0], [0, 0, len], [0.35, 0.55, 0.95]);
  return { vertices: new Float32Array(l), count: l.length / VERTEX_STRIDE };
}

export function createOctahedron(s: number): CpuMesh {
  const p: [number, number, number][] = [
    [s, 0, 0],
    [-s, 0, 0],
    [0, s, 0],
    [0, -s, 0],
    [0, 0, s],
    [0, 0, -s],
  ];
  const faces: [number, number, number][] = [
    [0, 2, 4],
    [4, 2, 1],
    [1, 2, 5],
    [5, 2, 0],
    [0, 4, 3],
    [4, 1, 3],
    [1, 5, 3],
    [5, 0, 3],
  ];
  const verts: number[] = [];
  const idx: number[] = [];
  let base = 0;
  for (const f of faces) {
    const a = p[f[0]]!, b = p[f[1]]!, c = p[f[2]]!;
    const nx = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]);
    const ny = (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]);
    const nz = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const l = Math.hypot(nx, ny, nz) || 1;
    const n: [number, number, number] = [nx / l, ny / l, nz / l];
    pushVert(verts, a, n, [0, 0]);
    pushVert(verts, b, n, [1, 0]);
    pushVert(verts, c, n, [0.5, 1]);
    idx.push(base, base + 1, base + 2);
    base += 3;
  }
  return pack(verts, idx);
}

export function createWedge(s: number, span: number): CpuMesh {
  const verts: number[] = [];
  const idx: number[] = [];
  const quad = (
    a: [number, number, number],
    b: [number, number, number],
    c: [number, number, number],
    d: [number, number, number],
  ) => {
    const nx = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]);
    const ny = (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]);
    const nz = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const ln = Math.hypot(nx, ny, nz) || 1;
    const nn: [number, number, number] = [nx / ln, ny / ln, nz / ln];
    const base = verts.length / VERTEX_STRIDE;
    pushVert(verts, a, nn, [0, 0]);
    pushVert(verts, b, nn, [1, 0]);
    pushVert(verts, c, nn, [1, 1]);
    pushVert(verts, d, nn, [0, 1]);
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  };
  const tri = (
    a: [number, number, number],
    b: [number, number, number],
    c: [number, number, number],
  ) => {
    const nx = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]);
    const ny = (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]);
    const nz = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const ln = Math.hypot(nx, ny, nz) || 1;
    const nn: [number, number, number] = [nx / ln, ny / ln, nz / ln];
    const base = verts.length / VERTEX_STRIDE;
    pushVert(verts, a, nn, [0, 0]);
    pushVert(verts, b, nn, [1, 0]);
    pushVert(verts, c, nn, [0.5, 1]);
    idx.push(base, base + 1, base + 2);
  };
  const hz = span / 2;
  const n0: [number, number, number] = [-s, 0, -hz];
  const n1: [number, number, number] = [-s, 0, hz];
  const u0: [number, number, number] = [s * 0.75, s * 0.7, -hz];
  const u1: [number, number, number] = [s * 0.75, s * 0.7, hz];
  const d0: [number, number, number] = [s * 0.75, -s * 0.7, -hz];
  const d1: [number, number, number] = [s * 0.75, -s * 0.7, hz];
  quad(n0, u0, u1, n1);
  quad(n1, d1, d0, n0);
  quad(u0, d0, d1, u1);
  tri(n0, d0, u0);
  tri(n1, u1, d1);
  return pack(verts, idx);
}

function nacaYt(x: number, t: number): number {
  const xt = Math.min(Math.max(x, 0), 1);
  return (
    5 *
    t *
    (0.2969 * Math.sqrt(xt) -
      0.126 * xt -
      0.3516 * xt * xt +
      0.2843 * xt * xt * xt -
      0.1036 * xt * xt * xt * xt)
  );
}

function nacaYc(x: number, m: number, p = 0.4): number {
  if (m < 1e-6) return 0;
  const xt = Math.min(Math.max(x, 0), 1);
  if (xt < p) return (m * (2 * p * xt - xt * xt)) / (p * p);
  return (m * (1 - 2 * p + 2 * p * xt - xt * xt)) / ((1 - p) * (1 - p));
}

/** Finite NACA wing. Quarter-chord at the origin, span along Z. */
export function createNacaWing(
  chord: number,
  span: number,
  thick = 0.12,
  camber = 0.02,
  cSegs = 28,
  sSegs = 12,
): CpuMesh {
  const verts: number[] = [];
  const idx: number[] = [];
  const rows: { p: [number, number, number]; uv: [number, number] }[][] = [];
  for (let j = 0; j <= sSegs; j++) {
    const v = j / sSegs;
    const z = (v - 0.5) * span;
    const row: { p: [number, number, number]; uv: [number, number] }[] = [];
    for (let i = 0; i <= cSegs; i++) {
      const u = i / cSegs;
      const yt = nacaYt(u, thick) * chord;
      const yc = nacaYc(u, camber) * chord;
      const x = (u - 0.25) * chord;
      row.push({ p: [x, yc + yt, z], uv: [u, v] });
    }
    for (let i = cSegs; i >= 0; i--) {
      const u = i / cSegs;
      const yt = nacaYt(u, thick) * chord;
      const yc = nacaYc(u, camber) * chord;
      const x = (u - 0.25) * chord;
      row.push({ p: [x, yc - yt, z], uv: [u, v] });
    }
    rows.push(row);
  }
  const stride = rows[0]!.length;
  for (let j = 0; j <= sSegs; j++) {
    for (const q of rows[j]!) pushVert(verts, q.p, [0, 1, 0], q.uv);
  }
  for (let j = 0; j < sSegs; j++) {
    for (let i = 0; i < stride - 1; i++) {
      const a = j * stride + i;
      const b = a + stride;
      idx.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  const nVert = verts.length / VERTEX_STRIDE;
  const acc = new Float32Array(nVert * 3);
  for (let t = 0; t < idx.length; t += 3) {
    const i0 = idx[t]!, i1 = idx[t + 1]!, i2 = idx[t + 2]!;
    const o0 = i0 * VERTEX_STRIDE, o1 = i1 * VERTEX_STRIDE, o2 = i2 * VERTEX_STRIDE;
    const ax = verts[o1]! - verts[o0]!, ay = verts[o1 + 1]! - verts[o0 + 1]!, az = verts[o1 + 2]! - verts[o0 + 2]!;
    const bx = verts[o2]! - verts[o0]!, by = verts[o2 + 1]! - verts[o0 + 1]!, bz = verts[o2 + 2]! - verts[o0 + 2]!;
    const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
    for (const ii of [i0, i1, i2]) {
      acc[ii * 3] = (acc[ii * 3] ?? 0) + nx;
      acc[ii * 3 + 1] = (acc[ii * 3 + 1] ?? 0) + ny;
      acc[ii * 3 + 2] = (acc[ii * 3 + 2] ?? 0) + nz;
    }
  }
  for (let i = 0; i < nVert; i++) {
    const nx = acc[i * 3] ?? 0, ny = acc[i * 3 + 1] ?? 1, nz = acc[i * 3 + 2] ?? 0;
    const l = Math.hypot(nx, ny, nz) || 1;
    const o = i * VERTEX_STRIDE;
    verts[o + 3] = nx / l;
    verts[o + 4] = ny / l;
    verts[o + 5] = nz / l;
  }
  return pack(verts, idx);
}

export function createTunnelWires(
  x0: number,
  x1: number,
  y: number,
  z: number,
): { vertices: Float32Array; count: number } {
  const l: number[] = [];
  const col: [number, number, number] = [0.28, 0.32, 0.3];
  const add = (a: [number, number, number], b: [number, number, number], c = col) => {
    pushVert(l, a, c, [0, 0]);
    pushVert(l, b, c, [1, 0]);
  };
  const rect = (x: number) => {
    add([x, 0, -z], [x, 0, z]);
    add([x, 0, z], [x, y, z]);
    add([x, y, z], [x, y, -z]);
    add([x, y, -z], [x, 0, -z]);
  };
  rect(x0);
  rect(x1);
  add([x0, 0, -z], [x1, 0, -z]);
  add([x0, 0, z], [x1, 0, z]);
  add([x0, y, -z], [x1, y, -z]);
  add([x0, y, z], [x1, y, z]);
  return { vertices: new Float32Array(l), count: l.length / VERTEX_STRIDE };
}

