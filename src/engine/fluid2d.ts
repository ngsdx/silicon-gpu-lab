/**
 * GPU 2D incompressible Navier–Stokes (stable fluids + no-slip bodies).
 * Ping-pong RGBA16F FBOs, semi-Lagrangian advection, Jacobi projection.
 * Same objects you'd allocate with glTexStorage2D / glFramebufferTexture2D
 * from a desktop 4.3 core profile.
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

export const BODY_NAMES = ["Cylinder", "Square", "Airfoil", "Wedge", "Plate", "Diamond"] as const;
export const VIZ_NAMES = ["Dye", "Vorticity", "Speed", "Pressure"] as const;

const GLSL_BODY = `
uniform float uShape, uAoA, uSize, uAspect;
uniform vec2 uBody;
vec2 toPhys(vec2 uv){ return vec2((uv.x - uBody.x) * uAspect, uv.y - uBody.y); }
vec2 rot2(vec2 p, float a){ float c=cos(a), s=sin(a); return vec2(c*p.x-s*p.y, s*p.x+c*p.y); }
float sdCircle(vec2 p, float r){ return length(p)-r; }
float sdBox(vec2 p, vec2 b){
  vec2 d = abs(p)-b;
  return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}
float sdTriangle(vec2 p, vec2 a, vec2 b, vec2 c){
  vec2 e0=b-a, e1=c-b, e2=a-c;
  vec2 v0=p-a, v1=p-b, v2=p-c;
  vec2 pq0=v0-e0*clamp(dot(v0,e0)/dot(e0,e0),0.0,1.0);
  vec2 pq1=v1-e1*clamp(dot(v1,e1)/dot(e1,e1),0.0,1.0);
  vec2 pq2=v2-e2*clamp(dot(v2,e2)/dot(e2,e2),0.0,1.0);
  float s = sign(e0.x*e2.y - e0.y*e2.x);
  vec2 d = min(min(vec2(dot(pq0,pq0), s*(v0.x*e0.y-v0.y*e0.x)),
                   vec2(dot(pq1,pq1), s*(v1.x*e1.y-v1.y*e1.x))),
                   vec2(dot(pq2,pq2), s*(v2.x*e2.y-v2.y*e2.x)));
  return -sqrt(d.x)*sign(d.y);
}
float sdNaca(vec2 p, float chord, float thick, float camber){
  vec2 q = p;
  q.x += chord * 0.22;
  float x = q.x / chord;
  float yt = 5.0 * thick * (0.2969*sqrt(clamp(x,0.0,1.0)) - 0.1260*x - 0.3516*x*x
    + 0.2843*x*x*x - 0.1036*x*x*x*x);
  float pp = 0.4;
  float yc = 0.0;
  if (camber > 0.0005 && x >= 0.0 && x <= 1.0) {
    yc = x < pp
      ? camber * (2.0*pp*x - x*x) / (pp*pp)
      : camber * ((1.0-2.0*pp) + 2.0*pp*x - x*x) / ((1.0-pp)*(1.0-pp));
  }
  float halfT = yt * chord;
  float camY = yc * chord;
  if (x < 0.0) return length(vec2(q.x, q.y - camY)) - halfT * 0.15;
  if (x > 1.0) return length(vec2(q.x - chord, q.y - camY));
  return abs(q.y - camY) - halfT;
}
float bodySdf(vec2 uv){
  vec2 p = toPhys(uv);
  float s = uSize;
  float a = uAoA;
  if (uShape < 0.5) return sdCircle(p, s);
  if (uShape < 1.5) return sdBox(rot2(p, a), vec2(s*0.92));
  if (uShape < 2.5) return sdNaca(rot2(p, a), s*3.45, 0.11, 0.04);
  if (uShape < 3.5) {
    vec2 q = rot2(p, a);
    return sdTriangle(q, vec2(-s*1.15,0.0), vec2(s*0.85,s*0.72), vec2(s*0.85,-s*0.72));
  }
  if (uShape < 4.5) return sdBox(rot2(p, a), vec2(s*1.35, s*0.07));
  return sdBox(rot2(p, 0.785398+a), vec2(s*0.78));
}
float occupied(vec2 uv){ return bodySdf(uv) < 0.0 ? 1.0 : 0.0; }
`;

const GLSL_SAMPLE = `
uniform vec2 uInvRes;
vec4 bilerp(sampler2D t, vec2 uv){
  vec2 res = 1.0 / uInvRes;
  vec2 p = uv * res - 0.5;
  vec2 f = fract(p);
  vec2 i = (floor(p) + 0.5) * uInvRes;
  vec4 a = texture(t, i);
  vec4 b = texture(t, i + vec2(uInvRes.x, 0.0));
  vec4 c = texture(t, i + vec2(0.0, uInvRes.y));
  vec4 d = texture(t, i + uInvRes);
  return mix(mix(a,b,f.x), mix(c,d,f.x), f.y);
}
`;

function fs(src: string): string {
  return `#version 300 es
precision highp float;
in vec2 vUv;
layout(location=0) out vec4 frag;
${GLSL_SAMPLE}
${GLSL_BODY}
${src}`;
}

type Pair = { read: Framebuffer; write: Framebuffer };

function makePair(gl: WebGL2RenderingContext, w: number, h: number): Pair {
  const nearest = gl.NEAREST;
  return {
    read: createFramebuffer(gl, w, h, [gl.RGBA16F], "none", nearest),
    write: createFramebuffer(gl, w, h, [gl.RGBA16F], "none", nearest),
  };
}

function swap(p: Pair) {
  const t = p.read;
  p.read = p.write;
  p.write = t;
}

function destroyPair(gl: WebGL2RenderingContext, p: Pair) {
  destroyFramebuffer(gl, p.read);
  destroyFramebuffer(gl, p.write);
}

export type FluidParams = {
  shape: number;
  aoa: number;
  size: number;
  bodyY: number;
  wind: number;
  visc: number;
  confine: number;
  viz: number;
  particles: number;
};

export class Fluid2D {
  readonly width: number;
  readonly height: number;
  readonly aspect: number;
  private vel: Pair;
  private prs: Pair;
  private dye: Pair;
  private tmp: Framebuffer;
  private parts: Pair;
  private progs: Record<string, ShaderProgram>;
  private body = { x: 0.28, y: 0.5, shape: 0, aoa: 0, size: 0.11 };

  constructor(
    private gl: WebGL2RenderingContext,
    private tri: GpuMesh,
    w: number,
    h: number,
  ) {
    const ext = gl.getExtension("EXT_color_buffer_float");
    if (!ext) throw new Error("EXT_color_buffer_float required for the wind-tunnel (RGBA16F FBOs).");
    this.width = w;
    this.height = h;
    this.aspect = w / h;
    this.vel = makePair(gl, w, h);
    this.prs = makePair(gl, w, h);
    this.dye = makePair(gl, w, h);
    this.tmp = createFramebuffer(gl, w, h, [gl.RGBA16F], "none", gl.NEAREST);
    this.parts = makePair(gl, 128, 48);
    this.progs = {
      advect: this.p(FS_ADVECT),
      confine: this.p(FS_CONFINE),
      divergence: this.p(FS_DIV),
      jacobi: this.p(FS_JACOBI),
      project: this.p(FS_PROJECT),
      boundary: this.p(FS_BOUNDARY),
      dye: this.p(FS_DYE),
      display: this.p(FS_DISPLAY),
      pstep: this.p(FS_PARTICLE_STEP),
      pdraw: createProgram(gl, VS_PARTICLE, FS_PARTICLE),
      clear: this.p(FS_CLEAR),
    };
    this.seedParticles();
  }

  private p(src: string) {
    return createProgram(this.gl, VS_BLIT, fs(src));
  }

  private seedParticles() {
    const gl = this.gl;
    const { width: w, height: h } = this.parts.read;
    bindFramebuffer(gl, this.parts.read);
    gl.useProgram(this.progs.clear.prog);
    gl.uniform4f(this.progs.clear.uniforms.uValue, -1, 0, 0, 0);
    this.drawTri();
    bindFramebuffer(gl, this.parts.write);
    gl.useProgram(this.progs.clear.prog);
    this.drawTri();
    bindFramebuffer(gl, null);
  }

  reset(wind: number) {
    const gl = this.gl;
    for (const fb of [this.vel.read, this.vel.write]) {
      bindFramebuffer(gl, fb);
      gl.useProgram(this.progs.clear.prog);
      gl.uniform4f(this.progs.clear.uniforms.uValue, wind, 0, 0, 0);
      this.drawTri();
    }
    for (const fb of [this.dye.read, this.dye.write, this.prs.read, this.prs.write]) {
      bindFramebuffer(gl, fb);
      gl.useProgram(this.progs.clear.prog);
      gl.uniform4f(this.progs.clear.uniforms.uValue, 0, 0, 0, 0);
      this.drawTri();
    }
    bindFramebuffer(gl, null);
  }

  private drawTri() {
    drawMesh(this.gl, this.tri);
  }

  private bindBody(p: ShaderProgram) {
    const gl = this.gl;
    gl.uniform1f(p.uniforms.uShape, this.body.shape);
    gl.uniform1f(p.uniforms.uAoA, this.body.aoa);
    gl.uniform1f(p.uniforms.uSize, this.body.size);
    gl.uniform1f(p.uniforms.uAspect, this.aspect);
    gl.uniform2f(p.uniforms.uBody, this.body.x, this.body.y);
    gl.uniform2f(p.uniforms.uInvRes, 1 / this.width, 1 / this.height);
  }

  private pass(p: ShaderProgram, target: Framebuffer, bind?: (p: ShaderProgram) => void) {
    const gl = this.gl;
    bindFramebuffer(gl, target);
    gl.useProgram(p.prog);
    this.bindBody(p);
    bind?.(p);
    this.drawTri();
  }

  step(dt: number, time: number, params: FluidParams) {
    this.body.shape = params.shape;
    this.body.aoa = params.aoa;
    this.body.size = params.size;
    this.body.y = params.bodyY;
    const gl = this.gl;
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.CULL_FACE);

    const wind = params.wind;
    const dtC = Math.min(dt, 0.033);

    this.pass(this.progs.advect, this.vel.write, (p) => {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.vel.read.colors[0]!);
      gl.uniform1i(p.uniforms.uVel, 0);
      gl.uniform1i(p.uniforms.uSrc, 0);
      gl.uniform1f(p.uniforms.uDt, dtC);
      gl.uniform1f(p.uniforms.uDissipate, Math.max(0.965, 1 - params.visc * 70));
    });
    swap(this.vel);

    if (params.confine > 0.001) {
      this.pass(this.progs.confine, this.vel.write, (p) => {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.vel.read.colors[0]!);
        gl.uniform1i(p.uniforms.uVel, 0);
        gl.uniform1f(p.uniforms.uEps, params.confine);
        gl.uniform1f(p.uniforms.uDt, dtC);
      });
      swap(this.vel);
    }

    this.pass(this.progs.boundary, this.vel.write, (p) => {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.vel.read.colors[0]!);
      gl.uniform1i(p.uniforms.uVel, 0);
      gl.uniform1f(p.uniforms.uWind, wind);
      gl.uniform1f(p.uniforms.uTime, time);
    });
    swap(this.vel);

    this.pass(this.progs.divergence, this.tmp, (p) => {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.vel.read.colors[0]!);
      gl.uniform1i(p.uniforms.uVel, 0);
    });

    const pIter = 12;
    for (let i = 0; i < pIter; i++) {
      this.pass(this.progs.jacobi, this.prs.write, (p) => {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.prs.read.colors[0]!);
        gl.uniform1i(p.uniforms.uX, 0);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.tmp.colors[0]!);
        gl.uniform1i(p.uniforms.uB, 1);
        gl.uniform1f(p.uniforms.uAlpha, -1);
        gl.uniform1f(p.uniforms.uBeta, 0.25);
        gl.uniform1f(p.uniforms.uPressure, 1);
      });
      swap(this.prs);
    }

    this.pass(this.progs.project, this.vel.write, (p) => {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.vel.read.colors[0]!);
      gl.uniform1i(p.uniforms.uVel, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.prs.read.colors[0]!);
      gl.uniform1i(p.uniforms.uPrs, 1);
    });
    swap(this.vel);

    this.pass(this.progs.boundary, this.vel.write, (p) => {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.vel.read.colors[0]!);
      gl.uniform1i(p.uniforms.uVel, 0);
      gl.uniform1f(p.uniforms.uWind, wind);
      gl.uniform1f(p.uniforms.uTime, time);
    });
    swap(this.vel);

    this.pass(this.progs.dye, this.dye.write, (p) => {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.vel.read.colors[0]!);
      gl.uniform1i(p.uniforms.uVel, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.dye.read.colors[0]!);
      gl.uniform1i(p.uniforms.uSrc, 1);
      gl.uniform1f(p.uniforms.uDt, dtC);
      gl.uniform1f(p.uniforms.uTime, time);
    });
    swap(this.dye);

    if (params.particles > 0.5) {
      bindFramebuffer(gl, this.parts.write);
      gl.viewport(0, 0, this.parts.write.width, this.parts.write.height);
      gl.useProgram(this.progs.pstep.prog);
      this.bindBody(this.progs.pstep);
      gl.uniform2f(this.progs.pstep.uniforms.uInvRes, 1 / this.width, 1 / this.height);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.vel.read.colors[0]!);
      gl.uniform1i(this.progs.pstep.uniforms.uVel, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.parts.read.colors[0]!);
      gl.uniform1i(this.progs.pstep.uniforms.uSrc, 1);
      gl.uniform1f(this.progs.pstep.uniforms.uDt, dtC);
      gl.uniform1f(this.progs.pstep.uniforms.uTime, time);
      this.drawTri();
      swap(this.parts);
    }
  }

  display(screenW: number, screenH: number, params: FluidParams, points: GpuMesh) {
    const gl = this.gl;
    bindFramebuffer(gl, null, screenW, screenH);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.useProgram(this.progs.display.prog);
    this.bindBody(this.progs.display);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.vel.read.colors[0]!);
    gl.uniform1i(this.progs.display.uniforms.uVel, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.dye.read.colors[0]!);
    gl.uniform1i(this.progs.display.uniforms.uDye, 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.prs.read.colors[0]!);
    gl.uniform1i(this.progs.display.uniforms.uPrs, 2);
    gl.uniform1f(this.progs.display.uniforms.uViz, params.viz);
    gl.uniform1f(this.progs.display.uniforms.uWind, params.wind);
    this.drawTri();

    if (params.particles > 0.5) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      gl.useProgram(this.progs.pdraw.prog);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.parts.read.colors[0]!);
      gl.uniform1i(this.progs.pdraw.uniforms.uParticles, 0);
      gl.uniform1f(this.progs.pdraw.uniforms.uSize, Math.max(1.6, screenH / 360));
      drawMesh(gl, points);
      gl.disable(gl.BLEND);
    }
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
  }

  dispose() {
    const gl = this.gl;
    destroyPair(gl, this.vel);
    destroyPair(gl, this.prs);
    destroyPair(gl, this.dye);
    destroyPair(gl, this.parts);
    destroyFramebuffer(gl, this.tmp);
    for (const p of Object.values(this.progs)) destroyProgram(gl, p);
  }
}

const FS_CLEAR = `
uniform vec4 uValue;
void main(){ frag = uValue; }
`;

const FS_ADVECT = `
uniform sampler2D uVel, uSrc;
uniform float uDt, uDissipate;
void main(){
  vec2 vel = bilerp(uVel, vUv).xy;
  vec2 uv = vUv - vel * uDt;
  uv = clamp(uv, uInvRes, 1.0 - uInvRes);
  frag = bilerp(uSrc, uv) * uDissipate;
}
`;

const FS_CONFINE = `
uniform sampler2D uVel;
uniform float uEps, uDt;
void main(){
  vec2 dx = vec2(uInvRes.x, 0.0);
  vec2 dy = vec2(0.0, uInvRes.y);
  vec2 vl = texture(uVel, vUv - dx).xy;
  vec2 vr = texture(uVel, vUv + dx).xy;
  vec2 vd = texture(uVel, vUv - dy).xy;
  vec2 vu = texture(uVel, vUv + dy).xy;
  float w = (vr.y - vl.y) * 0.5 / uInvRes.x - (vu.x - vd.x) * 0.5 / uInvRes.y;
  float wl = (texture(uVel, vUv - dx + dy).y - texture(uVel, vUv - dx - dy).y);
  float wr = (texture(uVel, vUv + dx + dy).y - texture(uVel, vUv + dx - dy).y);
  vec2 grad = vec2(abs(wr) - abs(wl), abs(vu.x) - abs(vd.x));
  float g = length(grad) + 1e-5;
  vec2 force = uEps * w * vec2(grad.y, -grad.x) / g;
  vec2 vel = texture(uVel, vUv).xy + force * uDt;
  frag = vec4(vel, 0.0, 1.0);
}
`;

const FS_DIV = `
uniform sampler2D uVel;
void main(){
  vec2 dx = vec2(uInvRes.x, 0.0);
  vec2 dy = vec2(0.0, uInvRes.y);
  float div = (texture(uVel, vUv+dx).x - texture(uVel, vUv-dx).x) * 0.5 / uInvRes.x
            + (texture(uVel, vUv+dy).y - texture(uVel, vUv-dy).y) * 0.5 / uInvRes.y;
  if (occupied(vUv) > 0.5) div = 0.0;
  frag = vec4(div, 0.0, 0.0, 1.0);
}
`;

const FS_JACOBI = `
uniform sampler2D uX, uB;
uniform float uAlpha, uBeta, uPressure;
void main(){
  vec2 dx = vec2(uInvRes.x, 0.0);
  vec2 dy = vec2(0.0, uInvRes.y);
  vec4 L = texture(uX, vUv - dx);
  vec4 R = texture(uX, vUv + dx);
  vec4 D = texture(uX, vUv - dy);
  vec4 U = texture(uX, vUv + dy);
  vec4 b = texture(uB, vUv);
  vec4 x = (L + R + D + U + uAlpha * b) * uBeta;
  if (uPressure > 0.5 && occupied(vUv) > 0.5) x = vec4(0.0);
  frag = x;
}
`;

const FS_PROJECT = `
uniform sampler2D uVel, uPrs;
void main(){
  vec2 dx = vec2(uInvRes.x, 0.0);
  vec2 dy = vec2(0.0, uInvRes.y);
  float pl = texture(uPrs, vUv - dx).x;
  float pr = texture(uPrs, vUv + dx).x;
  float pd = texture(uPrs, vUv - dy).x;
  float pu = texture(uPrs, vUv + dy).x;
  vec2 vel = texture(uVel, vUv).xy;
  vel.x -= (pr - pl) * 0.5 / uInvRes.x;
  vel.y -= (pu - pd) * 0.5 / uInvRes.y;
  frag = vec4(vel, 0.0, 1.0);
}
`;

const FS_BOUNDARY = `
uniform sampler2D uVel;
uniform float uWind, uTime;
void main(){
  vec2 vel = texture(uVel, vUv).xy;
  if (vUv.x < 2.0 * uInvRes.x) {
    vel = vec2(uWind, 0.02 * sin(uTime * 3.1 + vUv.y * 22.0) * uWind);
  }
  if (vUv.y < 2.0 * uInvRes.y || vUv.y > 1.0 - 2.0 * uInvRes.y) vel.y = 0.0;
  if (occupied(vUv) > 0.5) vel = vec2(0.0);
  frag = vec4(vel, 0.0, 1.0);
}
`;

const FS_DYE = `
uniform sampler2D uVel, uSrc;
uniform float uDt, uTime;
void main(){
  vec2 vel = bilerp(uVel, vUv).xy;
  vec2 uv = clamp(vUv - vel * uDt, uInvRes, 1.0 - uInvRes);
  vec3 c = bilerp(uSrc, uv).rgb * 0.996;
  if (vUv.x < 0.025) {
    float stripe = 0.5 + 0.5 * sin(vUv.y * 54.0);
    vec3 a = vec3(0.18, 0.72, 0.82);
    vec3 b = vec3(0.82, 0.55, 0.18);
    c = mix(a, b, stripe);
  }
  if (occupied(vUv) > 0.5) c = vec3(0.0);
  frag = vec4(c, 1.0);
}
`;

const FS_DISPLAY = `
uniform sampler2D uVel, uDye, uPrs;
uniform float uViz, uWind;
vec3 turbo(float x){
  x = clamp(x, 0.0, 1.0);
  return vec3(
    0.135 + 2.8*x - 6.2*x*x + 4.5*x*x*x,
    0.09 + 1.1*x + 0.4*sin(x*6.2),
    0.55 + 0.4*cos(x*5.0) - 0.85*x
  );
}
void main(){
  vec2 vel = texture(uVel, vUv).xy;
  float spd = length(vel);
  vec2 dx = vec2(uInvRes.x, 0.0);
  vec2 dy = vec2(0.0, uInvRes.y);
  float w = (texture(uVel, vUv+dx).y - texture(uVel, vUv-dx).y) * 0.5 / uInvRes.x
          - (texture(uVel, vUv+dy).x - texture(uVel, vUv-dy).x) * 0.5 / uInvRes.y;
  float pr = texture(uPrs, vUv).x;
  vec3 col;
  if (uViz < 0.5) {
    col = texture(uDye, vUv).rgb;
    col += vec3(0.04, 0.05, 0.06);
  } else if (uViz < 1.5) {
    float s = tanh(w * 0.018);
    col = vec3(0.08, 0.09, 0.1) + vec3(0.85, 0.18, 0.12) * max(s, 0.0) + vec3(0.15, 0.42, 0.95) * max(-s, 0.0);
  } else if (uViz < 2.5) {
    col = turbo(spd / max(uWind * 2.2, 0.05));
    col *= 0.85;
    col += 0.05;
  } else {
    float s = tanh(pr * 0.08);
    col = vec3(0.09) + vec3(0.82, 0.22, 0.16) * max(s,0.0) + vec3(0.18, 0.45, 0.95) * max(-s,0.0);
  }
  float occ = occupied(vUv);
  float edge = 0.0;
  edge += occupied(vUv + dx) + occupied(vUv - dx) + occupied(vUv + dy) + occupied(vUv - dy);
  if (occ > 0.5) col = vec3(0.11, 0.12, 0.125);
  else if (edge > 0.5) col = mix(col, vec3(0.78, 0.81, 0.79), 0.85);
  if (vUv.x < 0.018) {
    float chev = abs(fract(vUv.y * 14.0 + vUv.x * 8.0) - 0.5);
    col = mix(col, vec3(0.55, 0.62, 0.58), 1.0 - smoothstep(0.08, 0.22, chev));
  }
  frag = vec4(col, 1.0);
}
`;

const FS_PARTICLE_STEP = `
uniform sampler2D uVel, uSrc;
uniform float uDt, uTime;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void main(){
  vec4 s = texture(uSrc, vUv);
  vec2 p = s.xy;
  float age = s.z;
  if (p.x < 0.0 || p.x > 0.985 || p.y < 0.02 || p.y > 0.98 || occupied(p) > 0.5 || age > 9.0) {
    float y = 0.06 + 0.88 * hash(vUv + vec2(uTime * 0.01, 0.3));
    p = vec2(0.03 + 0.02 * hash(vUv * 3.1), y);
    age = 0.0;
  } else {
    vec2 vel = bilerp(uVel, p).xy;
    p += vel * uDt;
    age += uDt;
  }
  frag = vec4(p, age, 1.0);
}
`;

const VS_PARTICLE = `#version 300 es
layout(location=0) in vec2 aUv;
uniform sampler2D uParticles;
uniform float uSize;
out float vAge;
void main(){
  vec4 p = texture(uParticles, aUv);
  vAge = p.z;
  if (p.x < 0.0) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
    gl_PointSize = 0.0;
    return;
  }
  gl_Position = vec4(p.xy * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = uSize;
}`;

const FS_PARTICLE = `#version 300 es
precision highp float;
in float vAge;
out vec4 frag;
void main(){
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  float a = 1.0 - dot(q, q);
  if (a <= 0.0) discard;
  float fade = exp(-vAge * 0.18);
  frag = vec4(vec3(0.85, 0.92, 0.88) * fade, a * 0.55 * fade);
}`;
