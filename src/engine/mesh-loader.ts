/** Wavefront OBJ + binary/ASCII STL → triangle soup in a unit cube. */

export type LoadedMesh = {
  name: string;
  tris: Float32Array;
  triCount: number;
  ext: string;
};

const MAX_BYTES = 16 * 1024 * 1024;
const MAX_TRIS = 60_000;

export function extOf(name: string): string {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i + 1).toLowerCase() : "";
}

export function parseMeshFile(name: string, buffer: ArrayBuffer): LoadedMesh {
  if (buffer.byteLength > MAX_BYTES) throw new Error("File too large (max 16 MB).");
  const ext = extOf(name);
  let tris: Float32Array;
  if (ext === "obj") tris = parseObj(new TextDecoder().decode(buffer));
  else if (ext === "stl") tris = parseStl(buffer);
  else throw new Error("Use a Wavefront .obj or .stl file.");
  const triCount = tris.length / 9;
  if (triCount < 1) throw new Error("No triangles in that file.");
  if (triCount > MAX_TRIS) throw new Error(`Too many triangles (${triCount.toLocaleString()}, max ${MAX_TRIS.toLocaleString()}).`);
  normalizeInPlace(tris);
  return { name, tris, triCount, ext };
}

function parseObj(src: string): Float32Array {
  const verts: number[] = [];
  const faces: number[] = [];
  const lines = src.split(/\r?\n/);
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line[0] === "#") continue;
    if (line.startsWith("v ")) {
      const p = line.split(/\s+/);
      verts.push(Number(p[1]), Number(p[2]), Number(p[3]));
    } else if (line.startsWith("f ")) {
      const p = line.split(/\s+/).slice(1);
      const idx: number[] = [];
      for (const tok of p) {
        if (!tok) continue;
        const a = tok.split("/")[0]!;
        let i = Number(a);
        if (!Number.isFinite(i) || i === 0) continue;
        i = i < 0 ? verts.length / 3 + i : i - 1;
        idx.push(i);
      }
      for (let k = 1; k + 1 < idx.length; k++) {
        faces.push(idx[0]!, idx[k]!, idx[k + 1]!);
      }
    }
  }
  const n = (faces.length / 3) | 0;
  const tris = new Float32Array(n * 9);
  for (let t = 0; t < n; t++) {
    for (let k = 0; k < 3; k++) {
      const vi = faces[t * 3 + k]! * 3;
      tris[t * 9 + k * 3] = verts[vi] ?? 0;
      tris[t * 9 + k * 3 + 1] = verts[vi + 1] ?? 0;
      tris[t * 9 + k * 3 + 2] = verts[vi + 2] ?? 0;
    }
  }
  return tris;
}

function parseStl(buffer: ArrayBuffer): Float32Array {
  if (buffer.byteLength < 84) throw new Error("Truncated STL.");
  const view = new DataView(buffer);
  const n = view.getUint32(80, true);
  const binaryLen = 84 + n * 50;
  const head = new TextDecoder().decode(buffer.slice(0, 5)).toLowerCase();
  const looksBinary = binaryLen === buffer.byteLength || (n > 0 && n < 5e6 && binaryLen <= buffer.byteLength && head !== "solid");
  if (looksBinary && n > 0 && 84 + n * 50 <= buffer.byteLength) {
    const tris = new Float32Array(n * 9);
    let o = 84;
    for (let i = 0; i < n; i++) {
      // skip normal
      o += 12;
      for (let k = 0; k < 9; k++) {
        tris[i * 9 + k] = view.getFloat32(o, true);
        o += 4;
      }
      o += 2;
    }
    return tris;
  }
  return parseAsciiStl(new TextDecoder().decode(buffer));
}

function parseAsciiStl(src: string): Float32Array {
  const verts: number[] = [];
  const re = /vertex\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(src))) {
    verts.push(Number(m[1]), Number(m[2]), Number(m[3]));
  }
  if (verts.length < 9 || verts.length % 9 !== 0) throw new Error("Could not read ASCII STL.");
  return new Float32Array(verts);
}

function normalizeInPlace(tris: Float32Array) {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < tris.length; i += 3) {
    const x = tris[i]!, y = tris[i + 1]!, z = tris[i + 2]!;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const cz = (minZ + maxZ) * 0.5;
  const sx = Math.max(maxX - minX, 1e-6);
  const sy = Math.max(maxY - minY, 1e-6);
  const sz = Math.max(maxZ - minZ, 1e-6);
  const s = 2 / Math.max(sx, sy, sz);
  for (let i = 0; i < tris.length; i += 3) {
    tris[i] = (tris[i]! - cx) * s;
    tris[i + 1] = (tris[i + 1]! - cy) * s;
    tris[i + 2] = (tris[i + 2]! - cz) * s;
  }
}
