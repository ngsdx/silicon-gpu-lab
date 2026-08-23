/**
 * 3D aerodynamic velocity field (potential + vortex methods) and GPU tracers.
 * Sphere: exact doublet. Wing: lifting-line horseshoe. Bluff bodies: no-slip
 * SDF deflection + Kármán / hairpin shedding. Same GLSL is sampled by the
 * particle update pass and the body Cp shader.
 */
import { drawMesh, type GpuMesh } from "./buffer";
import { createProgram, destroyProgram, type ShaderProgram } from "./shader";
import { VS_BLIT } from "./common-gl";
import {
  bindFramebuffer,
  createFramebuffer,
  destroyFramebuffer,
  type Framebuffer,
} from "./framebuffer";

export const BODY3_NAMES = ["Sphere", "Cube", "Wing", "Wedge", "Plate", "Diamond"] as const;
export const VIZ3_NAMES = ["Streamlines", "Speed", "Swirl", "Pressure"] as const;

export const GLSL_FLOW = `
uniform float uShape, uAoA, uSize, uWind, uTime, uShed, uSpan;
uniform vec3 uBody;

vec3 toBody(vec3 p){
  vec3 q = p - uBody;
  float c = cos(-uAoA), s = sin(-uAoA);
  return vec3(c*q.x - s*q.y, s*q.x + c*q.y, q.z);
}
vec3 fromBodyV(vec3 v){
  float c = cos(uAoA), s = sin(uAoA);
  return vec3(c*v.x - s*v.y, s*v.x + c*v.y, v.z);
}
float sdSphere(vec3 p, float r){ return length(p) - r; }
float sdBox(vec3 p, vec3 b){
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}
float sdOcta(vec3 p, float s){
  p = abs(p);
  return (p.x + p.y + p.z - s) * 0.57735027;
}
float sdCylY(vec3 p, float r, float h){
  vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
float naca2(vec2 p, float chord, float thick, float camber){
  vec2 q = p;
  q.x += chord * 0.25;
  float x = q.x / chord;
  float yt = 5.0 * thick * (0.2969*sqrt(clamp(x,0.0,1.0)) - 0.1260*x - 0.3516*x*x
    + 0.2843*x*x*x - 0.1036*x*x*x*x);
  float pp = 0.4;
  float yc = 0.0;
  if (camber > 0.0004 && x >= 0.0 && x <= 1.0) {
    yc = x < pp
      ? camber * (2.0*pp*x - x*x) / (pp*pp)
      : camber * ((1.0-2.0*pp) + 2.0*pp*x - x*x) / ((1.0-pp)*(1.0-pp));
  }
  float halfT = yt * chord;
  float camY = yc * chord;
  if (x < 0.0) return length(vec2(q.x, q.y - camY)) - halfT * 0.2;
  if (x > 1.0) return length(vec2(q.x - chord, q.y - camY));
  return abs(q.y - camY) - halfT;
}
float sdWing(vec3 p, float chord, float span, float thick){
  float d2 = naca2(p.xy, chord, thick, 0.02);
  float dZ = abs(p.z) - span * 0.5;
  vec2 w = vec2(d2, dZ);
  return min(max(w.x, w.y), 0.0) + length(max(w, 0.0));
}
float sdWedge(vec3 p, float s, float span){
  vec3 a = vec3(-s, 0.0, 0.0);
  vec3 b = vec3(s*0.75, s*0.7, 0.0);
  vec3 c = vec3(s*0.75, -s*0.7, 0.0);
  vec3 pa = p - a, ba = b - a, ca = c - a;
  vec3 nor = normalize(cross(ba, ca));
  float d2 = abs(dot(pa, nor));
  vec3 q = p - nor * dot(pa, nor);
  vec3 v0 = b-a, v1 = c-a, v2 = q-a;
  float d00 = dot(v0,v0), d01 = dot(v0,v1), d11 = dot(v1,v1);
  float d20 = dot(v2,v0), d21 = dot(v2,v1);
  float inv = 1.0 / max(d00*d11 - d01*d01, 1e-6);
  float v = (d11*d20 - d01*d21) * inv;
  float w = (d00*d21 - d01*d20) * inv;
  float u = 1.0 - v - w;
  float inside = min(min(u, v), w);
  float tri = inside > 0.0 ? -d2 : d2 + length(max(vec2(-inside, 0.0), 0.0))*s;
  float dZ = abs(p.z) - span * 0.5;
  vec2 h = vec2(tri, dZ);
  return min(max(h.x, h.y), 0.0) + length(max(h, 0.0));
}
float localSdf(vec3 q){
  float s = uSize;
  if (uShape < 0.5) return sdSphere(q, s);
  if (uShape < 1.5) return sdBox(q, vec3(s));
  if (uShape < 2.5) return sdWing(q, s*1.7, uSpan, 0.12);
  if (uShape < 3.5) return sdWedge(q, s*1.15, uSpan * 0.72);
  if (uShape < 4.5) return sdBox(q, vec3(s*1.35, s*0.07, uSpan*0.5));
  return sdOcta(q, s*1.15);
}
float bodySdf(vec3 p){
  return localSdf(toBody(p));
}
vec3 bodyNrm(vec3 p){
  float e = 0.012;
  return normalize(vec3(
    bodySdf(p+vec3(e,0,0)) - bodySdf(p-vec3(e,0,0)),
    bodySdf(p+vec3(0,e,0)) - bodySdf(p-vec3(0,e,0)),
    bodySdf(p+vec3(0,0,e)) - bodySdf(p-vec3(0,0,e))
  ));
}
vec3 sphereDipole(vec3 p, float R, float U){
  float r = length(p);
  if (r < R * 1.001) return vec3(0.0);
  float r2 = r*r, r5 = r2*r2*r;
  float f = 0.5 * R*R*R * U / max(r5, 1e-8);
  return vec3(
    U + f * (2.0*p.x*p.x - p.y*p.y - p.z*p.z),
    f * (2.0*p.x*p.y),
    f * (2.0*p.x*p.z)
  );
}
vec3 vortexSeg(vec3 p, vec3 a, vec3 b, float gamma){
  vec3 r1 = p - a, r2 = p - b, r0 = b - a;
  float l1 = length(r1) + 1e-4, l2 = length(r2) + 1e-4;
  vec3 cr = cross(r1, r2);
  float cr2 = dot(cr, cr) + 0.018 * dot(r0, r0);
  float h = dot(r1/l1 - r2/l2, r0);
  return cr * (gamma * 0.07957747 * h / cr2);
}
vec3 horseshoe(vec3 p, float chord, float span, float U, float aoa){
  float stall = smoothstep(0.38, 0.22, abs(aoa));
  float Cl = clamp(6.28318 * aoa, -1.35, 1.35) * (0.35 + 0.65 * stall);
  float G = 0.5 * Cl * U * chord;
  float qc = -0.0;
  vec3 L = vec3(qc, 0.0, -span*0.5);
  vec3 R = vec3(qc, 0.0,  span*0.5);
  vec3 Ld = vec3(8.0, 0.0, -span*0.5);
  vec3 Rd = vec3(8.0, 0.0,  span*0.5);
  vec3 u = vortexSeg(p, L, R, G);
  u += vortexSeg(p, L, Ld, G);
  u += vortexSeg(p, Rd, R, G);
  return u;
}
vec3 karman(vec3 p, float t, float D, float U, float axisY){
  float St = 0.19;
  float w = St * U / max(D, 0.05) * 6.28318;
  float ph = t * w;
  float amp = 0.55 * D;
  vec3 u = vec3(0.0);
  for (int i = 0; i < 5; i++) {
    float fi = float(i);
    float x = D*1.3 + fi * D*1.15;
    float sgn = mod(fi, 2.0) < 0.5 ? 1.0 : -1.0;
    float y = sgn * amp * sin(ph - fi * 1.1);
    vec3 c = vec3(x, axisY + y, 0.0);
    vec3 r = p - c;
    float rad2 = dot(r, r) + 0.05*D*D;
    u += cross(vec3(0.0, 0.0, sgn), r) * (0.42 * U * D * uShed / rad2) * exp(-fi * 0.18);
  }
  return u;
}
vec3 flowBody(vec3 q, float t){
  float U = uWind;
  vec3 inf = vec3(U, 0.0, 0.0);
  float s = uSize;
  vec3 u;
  if (uShape < 0.5) {
    u = sphereDipole(q, s, U);
    u += karman(q, t, s*2.0, U, 0.0) * 0.45;
  } else if (uShape > 1.5 && uShape < 2.5) {
    u = inf + horseshoe(q, s*1.7, uSpan, U, uAoA);
    if (abs(uAoA) > 0.26) {
      float sep = smoothstep(0.22, 0.4, abs(uAoA));
      float behind = smoothstep(-0.1, 0.4, q.x);
      u += vec3(-0.2, 0.35 * sign(uAoA), 0.0) * U * sep * behind * exp(-q.z*q.z*0.5);
    }
  } else {
    float e = 0.015;
    vec3 n = normalize(vec3(
      localSdf(q+vec3(e,0,0)) - localSdf(q-vec3(e,0,0)),
      localSdf(q+vec3(0,e,0)) - localSdf(q-vec3(0,e,0)),
      localSdf(q+vec3(0,0,e)) - localSdf(q-vec3(0,0,e))
    ));
    float d = localSdf(q);
    float k = smoothstep(0.0, s * 2.4, d);
    vec3 slide = inf - n * dot(inf, n);
    u = mix(slide * 1.35, inf, k);
    float down = smoothstep(-0.05, 0.5, q.x);
    float wakeR = length(q.yz) / max(s*2.8, 0.1);
    float inWake = (1.0 - smoothstep(0.6, 1.4, wakeR)) * down * (1.0 - k);
    u = mix(u, vec3(-0.12*U, q.y*0.55, q.z*0.55), inWake * 0.85);
    u += karman(q, t, s*2.0, U, 0.0);
  }
  float d = localSdf(q);
  if (d < 0.02) u *= smoothstep(-0.02, 0.06, d);
  if (d < 0.0) u = vec3(0.0);
  return u;
}
vec3 flowWorld(vec3 p, float t){
  vec3 q = toBody(p);
  vec3 u = flowBody(q, t);
  u = fromBodyV(u);
  if (p.y < 0.04) u.y = max(u.y, 0.0);
  return u;
}
`;

export type FlowParams = {
  shape: number;
  aoa: number;
  size: number;
  bodyY: number;
  wind: number;
  shed: number;
  viz: number;
  particles: number;
};

type Pair = { read: Framebuffer; write: Framebuffer };

function makePair(gl: WebGL2RenderingContext, w: number, h: number): Pair {
  return {
    read: createFramebuffer(gl, w, h, [gl.RGBA16F], "none", gl.NEAREST),
    write: createFramebuffer(gl, w, h, [gl.RGBA16F], "none", gl.NEAREST),
  };
}

function swap(p: Pair) {
  const t = p.read;
  p.read = p.write;
  p.write = t;
}

const FS_PSTEP = `#version 300 es
precision highp float;
in vec2 vUv;
layout(location=0) out vec4 frag;
uniform sampler2D uSrc;
uniform float uDt;
${GLSL_FLOW}
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
vec3 spawn(vec2 uv){
  float hx = hash(uv);
  float hy = hash(uv + 3.1);
  float hz = hash(uv + 7.7);
  if (hx < 0.78) {
    float iy = floor(uv.x * 14.0);
    float iz = floor(uv.y * 10.0);
    return vec3(-5.4, 0.18 + iy * 0.2 + hy * 0.04, (iz - 4.5) * 0.32 + hz * 0.04);
  }
  return vec3(-5.4, 0.15 + hy * 2.4, (hz - 0.5) * 3.0);
}
void main(){
  vec4 s = texture(uSrc, vUv);
  vec3 p = s.xyz;
  float age = s.w;
  float d = bodySdf(p);
  if (p.x > 6.4 || p.x < -5.8 || p.y < 0.02 || p.y > 3.4 || abs(p.z) > 3.4 || d < 0.0 || age > 8.0) {
    p = spawn(vUv + vec2(uTime * 0.001, 0.0));
    age = 0.0;
  } else {
    vec3 k1 = flowWorld(p, uTime);
    vec3 k2 = flowWorld(p + k1 * uDt, uTime);
    p += 0.5 * (k1 + k2) * uDt;
    age += uDt;
  }
  frag = vec4(p, age);
}
`;

const FS_SEED = `#version 300 es
precision highp float;
in vec2 vUv;
layout(location=0) out vec4 frag;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
void main(){
  float hy = hash(vUv + 3.1);
  float hz = hash(vUv + 7.7);
  float iy = floor(vUv.x * 14.0);
  float iz = floor(vUv.y * 10.0);
  vec3 p = vec3(-5.4, 0.18 + iy * 0.2 + hy * 0.04, (iz - 4.5) * 0.32 + hz * 0.04);
  frag = vec4(p, hash(vUv) * 4.0);
}
`;

export const VS_TRACER = `#version 300 es
layout(location=0) in vec2 aUv;
uniform sampler2D uParticles;
uniform mat4 uView, uProj;
uniform float uViz, uPoint;
out vec3 vCol;
out float vAlpha;
${GLSL_FLOW}
vec3 turbo(float x){
  x = clamp(x, 0.0, 1.0);
  return vec3(0.14+2.6*x-5.8*x*x+4.2*x*x*x, 0.1+1.05*x+0.35*sin(x*6.2), 0.52+0.4*cos(x*5.0)-0.8*x);
}
void main(){
  vec4 s = texture(uParticles, aUv);
  vec3 p = s.xyz;
  float age = s.w;
  vec3 u = flowWorld(p, uTime);
  float spd = length(u);
  vec3 swirl = cross(u, vec3(uWind, 0.0, 0.0));
  if (uViz < 0.5) {
    float layer = clamp((p.y - 0.1) / 2.2, 0.0, 1.0);
    vCol = mix(vec3(0.18, 0.72, 0.82), vec3(0.86, 0.58, 0.2), layer);
  } else if (uViz < 1.5) {
    vCol = turbo(spd / max(uWind * 1.8, 0.2));
  } else if (uViz < 2.5) {
    float sw = tanh(length(swirl) * 1.4);
    vCol = mix(vec3(0.15, 0.35, 0.85), vec3(0.9, 0.25, 0.15), sw);
  } else {
    float cp = 1.0 - (spd*spd) / max(uWind*uWind, 0.05);
    vCol = mix(vec3(0.18, 0.42, 0.95), vec3(0.9, 0.22, 0.16), clamp(cp * 0.5 + 0.5, 0.0, 1.0));
  }
  vAlpha = exp(-age * 0.12) * 0.72;
  gl_Position = uProj * uView * vec4(p, 1.0);
  gl_PointSize = clamp(uPoint / max(gl_Position.w, 0.2), 1.4, 7.0);
}
`;

export const FS_TRACER = `#version 300 es
precision highp float;
in vec3 vCol;
in float vAlpha;
out vec4 frag;
void main(){
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  float a = 1.0 - dot(q, q);
  if (a <= 0.0) discard;
  frag = vec4(vCol, a * vAlpha);
}
`;

export class ParticleField {
  readonly pw: number;
  readonly ph: number;
  private pair: Pair;
  private stepProg: ShaderProgram;
  private seedProg: ShaderProgram;
  constructor(
    private gl: WebGL2RenderingContext,
    private tri: GpuMesh,
    pw: number,
    ph: number,
  ) {
    if (!gl.getExtension("EXT_color_buffer_float")) {
      throw new Error("EXT_color_buffer_float required for 3D tracers (RGBA16F).");
    }
    this.pw = pw;
    this.ph = ph;
    this.pair = makePair(gl, pw, ph);
    this.stepProg = createProgram(gl, VS_BLIT, FS_PSTEP);
    this.seedProg = createProgram(gl, VS_BLIT, FS_SEED);
    this.seed();
  }

  private seed() {
    const gl = this.gl;
    for (const fb of [this.pair.read, this.pair.write]) {
      bindFramebuffer(gl, fb);
      gl.useProgram(this.seedProg.prog);
      drawMesh(gl, this.tri);
    }
    bindFramebuffer(gl, null);
  }

  bindFlow(p: ShaderProgram, params: FlowParams, time: number) {
    const gl = this.gl;
    gl.uniform1f(p.uniforms.uShape, params.shape);
    gl.uniform1f(p.uniforms.uAoA, params.aoa);
    gl.uniform1f(p.uniforms.uSize, params.size);
    gl.uniform1f(p.uniforms.uWind, params.wind);
    gl.uniform1f(p.uniforms.uTime, time);
    gl.uniform1f(p.uniforms.uShed, params.shed);
    gl.uniform1f(p.uniforms.uSpan, params.size * 2.6);
    gl.uniform3f(p.uniforms.uBody, 0, params.bodyY, 0);
  }

  step(dt: number, time: number, params: FlowParams) {
    const gl = this.gl;
    bindFramebuffer(gl, this.pair.write);
    gl.viewport(0, 0, this.pw, this.ph);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.useProgram(this.stepProg.prog);
    this.bindFlow(this.stepProg, params, time);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.pair.read.colors[0]!);
    gl.uniform1i(this.stepProg.uniforms.uSrc, 0);
    gl.uniform1f(this.stepProg.uniforms.uDt, Math.min(dt, 0.033));
    drawMesh(gl, this.tri);
    swap(this.pair);
    bindFramebuffer(gl, null);
  }

  get tex() {
    return this.pair.read.colors[0]!;
  }

  dispose() {
    const gl = this.gl;
    destroyFramebuffer(gl, this.pair.read);
    destroyFramebuffer(gl, this.pair.write);
    destroyProgram(gl, this.stepProg);
    destroyProgram(gl, this.seedProg);
  }
}
