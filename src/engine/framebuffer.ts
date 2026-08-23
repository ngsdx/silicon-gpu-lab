export type Framebuffer = {
  fbo: WebGLFramebuffer;
  colors: WebGLTexture[];
  depth: WebGLTexture | WebGLRenderbuffer | null;
  width: number;
  height: number;
};

export function createColorTarget(
  gl: WebGL2RenderingContext,
  w: number,
  h: number,
  internal: number = WebGL2RenderingContext.RGBA8,
  filter: number = WebGL2RenderingContext.LINEAR,
): WebGLTexture {
  const tex = gl.createTexture();
  if (!tex) throw new Error("color target alloc failed");
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texStorage2D(gl.TEXTURE_2D, 1, internal, w, h);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

export function createDepthTexture(gl: WebGL2RenderingContext, w: number, h: number): WebGLTexture {
  const tex = gl.createTexture();
  if (!tex) throw new Error("depth tex alloc failed");
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.DEPTH_COMPONENT32F, w, h);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL);
  return tex;
}

export function createFramebuffer(
  gl: WebGL2RenderingContext,
  w: number,
  h: number,
  colorInternals: number[],
  depth: "texture" | "renderbuffer" | "none" = "renderbuffer",
  filter: number = WebGL2RenderingContext.LINEAR,
): Framebuffer {
  const fbo = gl.createFramebuffer();
  if (!fbo) throw new Error("fbo alloc failed");
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);

  const colors: WebGLTexture[] = [];
  const attachments: number[] = [];
  colorInternals.forEach((internal, i) => {
    const tex = createColorTarget(gl, w, h, internal, filter);
    const att = gl.COLOR_ATTACHMENT0 + i;
    gl.framebufferTexture2D(gl.FRAMEBUFFER, att, gl.TEXTURE_2D, tex, 0);
    colors.push(tex);
    attachments.push(att);
  });
  if (attachments.length) gl.drawBuffers(attachments as number[]);
  else gl.drawBuffers([gl.NONE]);

  let depthObj: WebGLTexture | WebGLRenderbuffer | null = null;
  if (depth === "texture") {
    const d = createDepthTexture(gl, w, h);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, d, 0);
    depthObj = d;
  } else if (depth === "renderbuffer") {
    const rb = gl.createRenderbuffer();
    if (!rb) throw new Error("rb alloc failed");
    gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH24_STENCIL8, w, h);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, rb);
    depthObj = rb;
  }

  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    throw new Error(`FBO incomplete 0x${status.toString(16)}`);
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { fbo, colors, depth: depthObj, width: w, height: h };
}

export function destroyFramebuffer(gl: WebGL2RenderingContext, fb: Framebuffer) {
  gl.deleteFramebuffer(fb.fbo);
  for (const c of fb.colors) gl.deleteTexture(c);
  if (fb.depth) {
    if (fb.depth instanceof WebGLTexture) gl.deleteTexture(fb.depth);
    else gl.deleteRenderbuffer(fb.depth);
  }
}

export function bindFramebuffer(gl: WebGL2RenderingContext, fb: Framebuffer | null, vw?: number, vh?: number) {
  if (!fb) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (vw && vh) gl.viewport(0, 0, vw, vh);
    return;
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb.fbo);
  gl.viewport(0, 0, fb.width, fb.height);
}
