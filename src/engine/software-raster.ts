/**
 * A CPU rasterizer that mirrors the GPU's fixed-function stages:
 *   vertex transform → perspective divide → viewport → barycentric
 *   fill (top-left) → perspective-correct attributes → depth test → ROP
 *
 * Slow on purpose. The point is to see the work a ROP/SIMD unit hides.
 */

export type SoftVertex = { x: number; y: number; z: number; r: number; g: number; b: number };
export type SoftTri = [SoftVertex, SoftVertex, SoftVertex];

export class SoftwareRaster {
  width: number;
  height: number;
  color: Uint8ClampedArray;
  depth: Float32Array;
  triangles = 0;
  pixelsFilled = 0;
  pixelsTested = 0;
  ms = 0;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.color = new Uint8ClampedArray(width * height * 4);
    this.depth = new Float32Array(width * height);
  }

  resize(width: number, height: number) {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    this.color = new Uint8ClampedArray(width * height * 4);
    this.depth = new Float32Array(width * height);
  }

  clear(r = 9, g = 10, b = 11) {
    this.color.fill(0);
    for (let i = 0; i < this.color.length; i += 4) {
      this.color[i] = r;
      this.color[i + 1] = g;
      this.color[i + 2] = b;
      this.color[i + 3] = 255;
    }
    this.depth.fill(1);
    this.triangles = 0;
    this.pixelsFilled = 0;
    this.pixelsTested = 0;
  }

  /**
   * `mvp` is column-major. Vertices are object-space, `rgb` per vertex 0-1.
   */
  drawIndexed(
    positions: Float32Array,
    strideFloats: number,
    indices: Uint32Array,
    mvp: Float32Array,
    shade: "vertex-color" | "bary" | "depth" | "normal" = "bary",
  ) {
    const t0 = performance.now();
    const w = this.width, h = this.height;
    const clip: number[] = [];
    const nV = positions.length / strideFloats;
    for (let i = 0; i < nV; i++) {
      const o = i * strideFloats;
      const x = positions[o]!, y = positions[o + 1]!, z = positions[o + 2]!;
      const nx = positions[o + 3] ?? 0, ny = positions[o + 4] ?? 1, nz = positions[o + 5] ?? 0;
      const cw =
        mvp[3]! * x + mvp[7]! * y + mvp[11]! * z + mvp[15]!;
      const cx =
        mvp[0]! * x + mvp[4]! * y + mvp[8]! * z + mvp[12]!;
      const cy =
        mvp[1]! * x + mvp[5]! * y + mvp[9]! * z + mvp[13]!;
      const cz =
        mvp[2]! * x + mvp[6]! * y + mvp[10]! * z + mvp[14]!;
      clip.push(cx, cy, cz, cw, nx, ny, nz);
    }

    for (let t = 0; t < indices.length; t += 3) {
      const ia = indices[t]! * 7, ib = indices[t + 1]! * 7, ic = indices[t + 2]! * 7;
      const aw = clip[ia + 3]!, bw = clip[ib + 3]!, cw = clip[ic + 3]!;
      if (aw <= 0 && bw <= 0 && cw <= 0) continue;

      const ax = (clip[ia]! / aw) * 0.5 + 0.5;
      const ay = (clip[ia + 1]! / aw) * 0.5 + 0.5;
      const bx = (clip[ib]! / bw) * 0.5 + 0.5;
      const by = (clip[ib + 1]! / bw) * 0.5 + 0.5;
      const cxn = (clip[ic]! / cw) * 0.5 + 0.5;
      const cyn = (clip[ic + 1]! / cw) * 0.5 + 0.5;

      const x0 = ax * w, y0 = (1 - ay) * h;
      const x1 = bx * w, y1 = (1 - by) * h;
      const x2 = cxn * w, y2 = (1 - cyn) * h;

      const area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
      // Y-down screen space flips winding: front faces (CCW in clip) are CW here.
      if (area >= 0) continue;

      const z0 = clip[ia + 2]! / aw;
      const z1 = clip[ib + 2]! / bw;
      const z2 = clip[ic + 2]! / cw;
      const invw0 = 1 / aw, invw1 = 1 / bw, invw2 = 1 / cw;

      const minX = Math.max(0, Math.floor(Math.min(x0, x1, x2)));
      const maxX = Math.min(w - 1, Math.ceil(Math.max(x0, x1, x2)));
      const minY = Math.max(0, Math.floor(Math.min(y0, y1, y2)));
      const maxY = Math.min(h - 1, Math.ceil(Math.max(y0, y1, y2)));

      this.triangles++;
      const n0x = clip[ia + 4]!, n0y = clip[ia + 5]!, n0z = clip[ia + 6]!;
      const n1x = clip[ib + 4]!, n1y = clip[ib + 5]!, n1z = clip[ib + 6]!;
      const n2x = clip[ic + 4]!, n2y = clip[ic + 5]!, n2z = clip[ic + 6]!;

      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          const px = x + 0.5, py = y + 0.5;
          const w0 = (x1 - px) * (y2 - py) - (x2 - px) * (y1 - py);
          const w1 = (x2 - px) * (y0 - py) - (x0 - px) * (y2 - py);
          const w2 = (x0 - px) * (y1 - py) - (x1 - px) * (y0 - py);
          if (w0 > 0 || w1 > 0 || w2 > 0) continue;
          this.pixelsTested++;
          const a = w0 / area, b = w1 / area, c = w2 / area;
          const invW = a * invw0 + b * invw1 + c * invw2;
          const z = (a * z0 * invw0 + b * z1 * invw1 + c * z2 * invw2) / invW;
          const ndcZ = z * 0.5 + 0.5;
          const di = y * w + x;
          if (ndcZ < 0 || ndcZ > 1 || ndcZ >= this.depth[di]!) continue;
          this.depth[di] = ndcZ;
          const ci = di * 4;
          let r = 0, g = 0, bl = 0;
          if (shade === "depth") {
            const d = 1 - ndcZ;
            r = g = bl = d * 255;
          } else if (shade === "normal") {
            const nx = (a * n0x * invw0 + b * n1x * invw1 + c * n2x * invw2) / invW;
            const ny = (a * n0y * invw0 + b * n1y * invw1 + c * n2y * invw2) / invW;
            const nz = (a * n0z * invw0 + b * n1z * invw1 + c * n2z * invw2) / invW;
            r = (nx * 0.5 + 0.5) * 255;
            g = (ny * 0.5 + 0.5) * 255;
            bl = (nz * 0.5 + 0.5) * 255;
          } else if (shade === "vertex-color") {
            r = (a * 1 + b * 0 + c * 0) * 255;
            g = (a * 0 + b * 1 + c * 0) * 255;
            bl = (a * 0 + b * 0 + c * 1) * 255;
          } else {
            r = a * 220 + 20;
            g = b * 200 + 30;
            bl = c * 230 + 24;
          }
          this.color[ci] = r;
          this.color[ci + 1] = g;
          this.color[ci + 2] = bl;
          this.color[ci + 3] = 255;
          this.pixelsFilled++;
        }
      }
    }
    this.ms = performance.now() - t0;
  }

  depthPreview(): Uint8ClampedArray {
    const out = new Uint8ClampedArray(this.color.length);
    for (let i = 0; i < this.depth.length; i++) {
      const d = (1 - this.depth[i]!) * 255;
      const o = i * 4;
      out[o] = d;
      out[o + 1] = d;
      out[o + 2] = d;
      out[o + 3] = 255;
    }
    return out;
  }
}
