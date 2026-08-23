/** CPU 3D signed-distance bake from a triangle soup. Uploaded as a TEXTURE_3D. */

export const SDF_N = 40;

function triDist2(
  px: number, py: number, pz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
): number {
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) return apx * apx + apy * apy + apz * apz;
  const bpx = px - bx, bpy = py - by, bpz = pz - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) return bpx * bpx + bpy * bpy + bpz * bpz;
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    const qx = apx - abx * v, qy = apy - aby * v, qz = apz - abz * v;
    return qx * qx + qy * qy + qz * qz;
  }
  const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) return cpx * cpx + cpy * cpy + cpz * cpz;
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    const qx = apx - acx * w, qy = apy - acy * w, qz = apz - acz * w;
    return qx * qx + qy * qy + qz * qz;
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
    const qx = bpx + (cx - bx) * w, qy = bpy + (cy - by) * w, qz = bpz + (cz - bz) * w;
    return qx * qx + qy * qy + qz * qz;
  }
  const denom = 1 / (va + vb + vc);
  const v = vb * denom;
  const w = vc * denom;
  const qx = apx - abx * v - acx * w;
  const qy = apy - aby * v - acy * w;
  const qz = apz - abz * v - acz * w;
  return qx * qx + qy * qy + qz * qz;
}

export function bakeSdf(tris: Float32Array, N = SDF_N): Float32Array {
  const pad = 1.15;
  const origin = -pad;
  const cell = (2 * pad) / N;
  const occ = new Uint8Array(N * N * N);
  const nTri = (tris.length / 9) | 0;
  const thresh2 = (cell * 1.05) * (cell * 1.05);

  for (let t = 0; t < nTri; t++) {
    const o = t * 9;
    const ax = tris[o]!, ay = tris[o + 1]!, az = tris[o + 2]!;
    const bx = tris[o + 3]!, by = tris[o + 4]!, bz = tris[o + 5]!;
    const cx = tris[o + 6]!, cy = tris[o + 7]!, cz = tris[o + 8]!;
    const minx = Math.max(0, Math.floor((Math.min(ax, bx, cx) - origin) / cell) - 1);
    const miny = Math.max(0, Math.floor((Math.min(ay, by, cy) - origin) / cell) - 1);
    const minz = Math.max(0, Math.floor((Math.min(az, bz, cz) - origin) / cell) - 1);
    const maxx = Math.min(N - 1, Math.ceil((Math.max(ax, bx, cx) - origin) / cell) + 1);
    const maxy = Math.min(N - 1, Math.ceil((Math.max(ay, by, cy) - origin) / cell) + 1);
    const maxz = Math.min(N - 1, Math.ceil((Math.max(az, bz, cz) - origin) / cell) + 1);
    for (let z = minz; z <= maxz; z++) {
      const pz = origin + (z + 0.5) * cell;
      for (let y = miny; y <= maxy; y++) {
        const py = origin + (y + 0.5) * cell;
        for (let x = minx; x <= maxx; x++) {
          const px = origin + (x + 0.5) * cell;
          if (triDist2(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz) <= thresh2) {
            occ[x + N * (y + N * z)] = 1;
          }
        }
      }
    }
  }

  // 1-voxel dilate so thin plates register as solid.
  const dil = new Uint8Array(occ);
  for (let z = 1; z < N - 1; z++) {
    for (let y = 1; y < N - 1; y++) {
      for (let x = 1; x < N - 1; x++) {
        if (occ[x + N * (y + N * z)]) continue;
        if (
          occ[x - 1 + N * (y + N * z)] ||
          occ[x + 1 + N * (y + N * z)] ||
          occ[x + N * (y - 1 + N * z)] ||
          occ[x + N * (y + 1 + N * z)] ||
          occ[x + N * (y + N * (z - 1))] ||
          occ[x + N * (y + N * (z + 1))]
        ) {
          dil[x + N * (y + N * z)] = 1;
        }
      }
    }
  }

  const dist = chamfer(dil, N, cell);
  const outside = floodOutside(dil, N);
  const sdf = new Float32Array(N * N * N);
  for (let i = 0; i < sdf.length; i++) {
    const s = outside[i] ? 1 : -1;
    sdf[i] = (dil[i] ? Math.min(dist[i]!, cell * 0.5) : dist[i]!) * s;
  }
  return sdf;
}

function chamfer(occ: Uint8Array, N: number, cell: number): Float32Array {
  const d = new Float32Array(N * N * N);
  d.fill(1e6);
  for (let i = 0; i < occ.length; i++) if (occ[i]) d[i] = 0;
  const nbs: [number, number, number, number][] = [
    [1, 0, 0, cell],
    [-1, 0, 0, cell],
    [0, 1, 0, cell],
    [0, -1, 0, cell],
    [0, 0, 1, cell],
    [0, 0, -1, cell],
    [1, 1, 0, cell * 1.414],
    [1, -1, 0, cell * 1.414],
    [-1, 1, 0, cell * 1.414],
    [-1, -1, 0, cell * 1.414],
    [1, 0, 1, cell * 1.414],
    [1, 0, -1, cell * 1.414],
    [-1, 0, 1, cell * 1.414],
    [-1, 0, -1, cell * 1.414],
    [0, 1, 1, cell * 1.414],
    [0, 1, -1, cell * 1.414],
    [0, -1, 1, cell * 1.414],
    [0, -1, -1, cell * 1.414],
  ];
  const sweep = (fwd: boolean) => {
    const a = fwd ? 0 : N - 1;
    const b = fwd ? N : -1;
    const s = fwd ? 1 : -1;
    for (let z = a; z !== b; z += s) {
      for (let y = a; y !== b; y += s) {
        for (let x = a; x !== b; x += s) {
          const i = x + N * (y + N * z);
          let best = d[i]!;
          for (const [dx, dy, dz, w] of nbs) {
            const xx = x + dx, yy = y + dy, zz = z + dz;
            if (xx < 0 || yy < 0 || zz < 0 || xx >= N || yy >= N || zz >= N) continue;
            const v = d[xx + N * (yy + N * zz)]! + w;
            if (v < best) best = v;
          }
          d[i] = best;
        }
      }
    }
  };
  sweep(true);
  sweep(false);
  return d;
}

function floodOutside(occ: Uint8Array, N: number): Uint8Array {
  const out = new Uint8Array(N * N * N);
  const q = new Int32Array(N * N * N);
  let head = 0, tail = 0;
  const push = (x: number, y: number, z: number) => {
    const i = x + N * (y + N * z);
    if (occ[i] || out[i]) return;
    out[i] = 1;
    q[tail++] = i;
  };
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      push(x, y, 0);
      push(x, y, N - 1);
    }
  }
  for (let z = 0; z < N; z++) {
    for (let x = 0; x < N; x++) {
      push(x, 0, z);
      push(x, N - 1, z);
    }
    for (let y = 0; y < N; y++) {
      push(0, y, z);
      push(N - 1, y, z);
    }
  }
  const dirs = [1, -1, N, -N, N * N, -N * N];
  while (head < tail) {
    const i = q[head++]!;
    const x = i % N;
    const y = Math.floor(i / N) % N;
    const z = Math.floor(i / (N * N));
    for (const ds of dirs) {
      const j = i + ds;
      if (j < 0 || j >= occ.length) continue;
      if (ds === 1 && x === N - 1) continue;
      if (ds === -1 && x === 0) continue;
      if (ds === N && y === N - 1) continue;
      if (ds === -N && y === 0) continue;
      if (ds === N * N && z === N - 1) continue;
      if (ds === -N * N && z === 0) continue;
      if (occ[j] || out[j]) continue;
      out[j] = 1;
      q[tail++] = j;
    }
  }
  return out;
}

export function createSdfTexture(gl: WebGL2RenderingContext, sdf: Float32Array, N: number): WebGLTexture {
  const tex = gl.createTexture();
  if (!tex) throw new Error("3D SDF alloc failed");
  gl.bindTexture(gl.TEXTURE_3D, tex);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.R32F, N, N, N, 0, gl.RED, gl.FLOAT, sdf);
  gl.bindTexture(gl.TEXTURE_3D, null);
  return tex;
}

export function createDummySdfTexture(gl: WebGL2RenderingContext): WebGLTexture {
  const sdf = new Float32Array(8);
  sdf.fill(1);
  const tex = gl.createTexture();
  if (!tex) throw new Error("dummy 3D SDF alloc failed");
  gl.bindTexture(gl.TEXTURE_3D, tex);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.R32F, 2, 2, 2, 0, gl.RED, gl.FLOAT, sdf);
  gl.bindTexture(gl.TEXTURE_3D, null);
  return tex;
}
