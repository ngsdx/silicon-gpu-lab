/**
 * Hardware interrogation — the kind of probe a driver engineer runs
 * before trusting a context. Maps 1:1 onto glGetString / glGetIntegerv
 * on desktop OpenGL.
 */

export type GpuLimits = {
  maxTextureSize: number;
  maxCubeMapSize: number;
  max3DTextureSize: number;
  maxArrayTextureLayers: number;
  maxRenderbufferSize: number;
  maxVertexAttribs: number;
  maxVertexUniformVectors: number;
  maxFragmentUniformVectors: number;
  maxVaryingVectors: number;
  maxTextureUnits: number;
  maxVertexTextureUnits: number;
  maxCombinedTextureUnits: number;
  maxDrawBuffers: number;
  maxColorAttachments: number;
  maxSamples: number;
  maxUniformBufferSize: number;
  maxUniformBufferBindings: number;
  uniformBufferOffsetAlignment: number;
  maxTransformFeedbackSeparateAttribs: number;
  maxElementIndex: number;
  aliasedLineWidth: [number, number];
  aliasedPointSize: [number, number];
  maxAnisotropy: number;
};

export type GpuCaps = {
  floatTextures: boolean;
  colorBufferFloat: boolean;
  colorBufferHalfFloat: boolean;
  depthTexture: boolean;
  instancedArrays: boolean;
  vertexArrayObject: boolean;
  drawBuffers: boolean;
  timerQuery: boolean;
  parallelShaderCompile: boolean;
  anisotropic: boolean;
  textureFloatLinear: boolean;
};

export type GpuInfo = {
  vendor: string;
  renderer: string;
  unmaskedVendor: string;
  unmaskedRenderer: string;
  version: string;
  shadingLanguage: string;
  limits: GpuLimits;
  caps: GpuCaps;
  extensions: string[];
  highpVertex: WebGLShaderPrecisionFormat | null;
  highpFragment: WebGLShaderPrecisionFormat | null;
};

export type GlContext = {
  gl: WebGL2RenderingContext;
  canvas: HTMLCanvasElement;
  info: GpuInfo;
  timerExt: EXTDisjointTimerQueryWebGL2 | null;
  anisotropyExt: EXT_texture_filter_anisotropic | null;
};

type EXTDisjointTimerQueryWebGL2 = {
  TIME_ELAPSED_EXT: number;
  GPU_DISJOINT_EXT: number;
  QUERY_RESULT_EXT: number;
  QUERY_RESULT_AVAILABLE_EXT: number;
  createQueryEXT?: () => WebGLQuery;
};

function precision(
  gl: WebGL2RenderingContext,
  shader: number,
  prec: number,
): WebGLShaderPrecisionFormat | null {
  return gl.getShaderPrecisionFormat(shader, prec);
}

export function createGlContext(canvas: HTMLCanvasElement): GlContext {
  const gl = canvas.getContext("webgl2", {
    alpha: false,
    antialias: true,
    depth: true,
    stencil: true,
    powerPreference: "high-performance",
    preserveDrawingBuffer: false,
    premultipliedAlpha: false,
  });
  if (!gl) {
    throw new Error("WebGL2 is required. This GPU/browser has no OpenGL ES 3.0 context.");
  }

  const debug = gl.getExtension("WEBGL_debug_renderer_info");
  const aniso = gl.getExtension("EXT_texture_filter_anisotropic");
  const timer = gl.getExtension("EXT_disjoint_timer_query_webgl2") as EXTDisjointTimerQueryWebGL2 | null;

  const extList = gl.getSupportedExtensions() ?? [];

  const info: GpuInfo = {
    vendor: gl.getParameter(gl.VENDOR) as string,
    renderer: gl.getParameter(gl.RENDERER) as string,
    unmaskedVendor: debug ? (gl.getParameter(debug.UNMASKED_VENDOR_WEBGL) as string) : "hidden",
    unmaskedRenderer: debug ? (gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) as string) : "hidden",
    version: gl.getParameter(gl.VERSION) as string,
    shadingLanguage: gl.getParameter(gl.SHADING_LANGUAGE_VERSION) as string,
    extensions: extList,
    highpVertex: precision(gl, gl.VERTEX_SHADER, gl.HIGH_FLOAT),
    highpFragment: precision(gl, gl.FRAGMENT_SHADER, gl.HIGH_FLOAT),
    limits: {
      maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
      maxCubeMapSize: gl.getParameter(gl.MAX_CUBE_MAP_TEXTURE_SIZE) as number,
      max3DTextureSize: gl.getParameter(gl.MAX_3D_TEXTURE_SIZE) as number,
      maxArrayTextureLayers: gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number,
      maxRenderbufferSize: gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) as number,
      maxVertexAttribs: gl.getParameter(gl.MAX_VERTEX_ATTRIBS) as number,
      maxVertexUniformVectors: gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS) as number,
      maxFragmentUniformVectors: gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS) as number,
      maxVaryingVectors: gl.getParameter(gl.MAX_VARYING_VECTORS) as number,
      maxTextureUnits: gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS) as number,
      maxVertexTextureUnits: gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS) as number,
      maxCombinedTextureUnits: gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS) as number,
      maxDrawBuffers: gl.getParameter(gl.MAX_DRAW_BUFFERS) as number,
      maxColorAttachments: gl.getParameter(gl.MAX_COLOR_ATTACHMENTS) as number,
      maxSamples: gl.getParameter(gl.MAX_SAMPLES) as number,
      maxUniformBufferSize: gl.getParameter(gl.MAX_UNIFORM_BLOCK_SIZE) as number,
      maxUniformBufferBindings: gl.getParameter(gl.MAX_UNIFORM_BUFFER_BINDINGS) as number,
      uniformBufferOffsetAlignment: gl.getParameter(gl.UNIFORM_BUFFER_OFFSET_ALIGNMENT) as number,
      maxTransformFeedbackSeparateAttribs: gl.getParameter(
        gl.MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS,
      ) as number,
      maxElementIndex: gl.getParameter(gl.MAX_ELEMENT_INDEX) as number,
      aliasedLineWidth: gl.getParameter(gl.ALIASED_LINE_WIDTH_RANGE) as [number, number],
      aliasedPointSize: gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE) as [number, number],
      maxAnisotropy: aniso ? (gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT) as number) : 1,
    },
    caps: {
      floatTextures: extList.includes("EXT_color_buffer_float") || extList.includes("OES_texture_float"),
      colorBufferFloat: !!gl.getExtension("EXT_color_buffer_float"),
      colorBufferHalfFloat: !!gl.getExtension("EXT_color_buffer_half_float"),
      depthTexture: true,
      instancedArrays: true,
      vertexArrayObject: true,
      drawBuffers: true,
      timerQuery: !!timer,
      parallelShaderCompile: !!gl.getExtension("KHR_parallel_shader_compile"),
      anisotropic: !!aniso,
      textureFloatLinear: !!gl.getExtension("OES_texture_float_linear"),
    },
  };

  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  gl.enable(gl.CULL_FACE);
  gl.cullFace(gl.BACK);
  gl.frontFace(gl.CCW);
  gl.clearColor(0.035, 0.039, 0.043, 1);

  return { gl, canvas, info, timerExt: timer, anisotropyExt: aniso };
}

export function resizeCanvasToDisplay(canvas: HTMLCanvasElement, maxDpr = 2): boolean {
  const dpr = Math.min(window.devicePixelRatio || 1, maxDpr);
  const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    return true;
  }
  return false;
}

export function gpuShortName(info: GpuInfo): string {
  const raw = info.unmaskedRenderer !== "hidden" ? info.unmaskedRenderer : info.renderer;
  return raw.replace(/ANGLE \((.+)\)/, "$1").slice(0, 64);
}
