import { hash01 } from "./math";

export function createTexture2D(
  gl: WebGL2RenderingContext,
  width: number,
  height: number,
  data: Uint8Array | Uint8ClampedArray | null,
  opts: {
    wrap?: number;
    mag?: number;
    min?: number;
    anisotropy?: number;
    internal?: number;
    format?: number;
    type?: number;
    mipmap?: boolean;
  } = {},
): WebGLTexture {
  const tex = gl.createTexture();
  if (!tex) throw new Error("texture alloc failed");
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    opts.internal ?? gl.RGBA,
    width,
    height,
    0,
    opts.format ?? gl.RGBA,
    opts.type ?? gl.UNSIGNED_BYTE,
    data,
  );
  const wrap = opts.wrap ?? gl.REPEAT;
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, opts.mag ?? gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, opts.min ?? gl.LINEAR_MIPMAP_LINEAR);
  if (opts.mipmap !== false && data) gl.generateMipmap(gl.TEXTURE_2D);
  if (opts.anisotropy && opts.anisotropy > 1) {
    const ext = gl.getExtension("EXT_texture_filter_anisotropic");
    if (ext) gl.texParameterf(gl.TEXTURE_2D, ext.TEXTURE_MAX_ANISOTROPY_EXT, opts.anisotropy);
  }
  return tex;
}

export function bindTexture(gl: WebGL2RenderingContext, unit: number, tex: WebGLTexture) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
}

export function makeChecker(size = 256, cells = 8): Uint8Array {
  const data = new Uint8Array(size * size * 4);
  const cs = size / cells;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const c = ((Math.floor(x / cs) + Math.floor(y / cs)) & 1) === 0;
      const i = (y * size + x) * 4;
      const lo = c ? 232 : 28;
      const hi = c ? 236 : 34;
      data[i] = lo;
      data[i + 1] = hi;
      data[i + 2] = c ? 228 : 38;
      data[i + 3] = 255;
    }
  }
  return data;
}

export function makeMarble(size = 256): Uint8Array {
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const nx = x / size, ny = y / size;
      const n =
        Math.sin((nx * 6 + Math.sin(ny * 9) * 0.4) * Math.PI * 2) * 0.5 +
        0.5 +
        (hash01(x * 13.1 + y * 7.7) - 0.5) * 0.08;
      const v = Math.pow(Math.abs(n * 2 - 1), 0.45);
      const i = (y * size + x) * 4;
      data[i] = 40 + v * 180;
      data[i + 1] = 42 + v * 176;
      data[i + 2] = 48 + v * 168;
      data[i + 3] = 255;
    }
  }
  return data;
}

export function makeBrickAlbedo(size = 256): Uint8Array {
  const data = new Uint8Array(size * size * 4);
  const bw = size / 4, bh = size / 8;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const row = Math.floor(y / bh);
      const ox = row & 1 ? bw / 2 : 0;
      const mx = (x + ox) % bw;
      const my = y % bh;
      const mortar = mx < 3 || my < 3;
      const n = hash01(Math.floor((x + ox) / bw) * 19 + row * 47);
      const i = (y * size + x) * 4;
      if (mortar) {
        data[i] = 196;
        data[i + 1] = 192;
        data[i + 2] = 184;
      } else {
        data[i] = 118 + n * 40;
        data[i + 1] = 62 + n * 20;
        data[i + 2] = 48 + n * 12;
      }
      data[i + 3] = 255;
    }
  }
  return data;
}

export function makeBrickHeight(size = 256): Float32Array {
  const h = new Float32Array(size * size);
  const bw = size / 4, bh = size / 8;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const row = Math.floor(y / bh);
      const ox = row & 1 ? bw / 2 : 0;
      const mx = (x + ox) % bw;
      const my = y % bh;
      const mortar = mx < 3 || my < 3;
      h[y * size + x] = mortar ? 0 : 1;
    }
  }
  return h;
}

export function heightToNormal(height: Float32Array, size: number, strength = 4): Uint8Array {
  const data = new Uint8Array(size * size * 4);
  const at = (x: number, y: number) => height[((y + size) % size) * size + ((x + size) % size)]!;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (at(x + 1, y) - at(x - 1, y)) * strength;
      const dy = (at(x, y + 1) - at(x, y - 1)) * strength;
      let nx = -dx, ny = -dy, nz = 1;
      const l = Math.hypot(nx, ny, nz) || 1;
      nx /= l;
      ny /= l;
      nz /= l;
      const i = (y * size + x) * 4;
      data[i] = (nx * 0.5 + 0.5) * 255;
      data[i + 1] = (ny * 0.5 + 0.5) * 255;
      data[i + 2] = (nz * 0.5 + 0.5) * 255;
      data[i + 3] = 255;
    }
  }
  return data;
}
