import { createMesh, createRawMesh, drawMesh, type GpuMesh } from "@/engine/buffer";
import { createProgram } from "@/engine/shader";
import {
  disposeCommon,
  drawGrid,
  makeFullscreenTri,
  makeGrid,
  makeLineProgram,
  setCameraUniforms,
  uploadMesh,
} from "@/engine/common-gl";
import {
  createCube,
  createIcosphere,
  createNacaWing,
  createOctahedron,
  createPlane,
  createTunnelWires,
  createWedge,
} from "@/engine/mesh";
import {
  BODY3_NAMES,
  FS_TRACER,
  GLSL_FLOW,
  ParticleField,
  VIZ3_NAMES,
  VS_TRACER,
  type FlowParams,
} from "@/engine/flow3d";
import {
  mat4FromRotationZ,
  mat4FromScaling,
  mat4FromTranslation,
  mat4Identity,
  mat4Multiply,
  mat4Normal,
} from "@/engine/math";
import type { Camera } from "@/engine/camera";
import type { LabDefinition, LabInstance } from "@/engine/lab-types";
import type { GlContext } from "@/engine/gpu";

const PW = 128;
const PH = 80;

function particleIds(gl: WebGL2RenderingContext): GpuMesh {
  const n = PW * PH;
  const uv = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    uv[i * 2] = ((i % PW) + 0.5) / PW;
    uv[i * 2 + 1] = (Math.floor(i / PW) + 0.5) / PH;
  }
  return createRawMesh(gl, uv, 2, gl.POINTS);
}

function pack(params: Record<string, number>): FlowParams {
  return {
    shape: Math.round(params.body ?? 2),
    aoa: ((params.aoa ?? 8) * Math.PI) / 180,
    size: params.size ?? 0.55,
    bodyY: params.height ?? 0.72,
    wind: params.wind ?? 1.65,
    shed: params.shed ?? 0.7,
    viz: Math.round(params.viz ?? 0),
    particles: params.particles ?? 1,
  };
}

const VS_BODY = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
uniform mat4 uModel, uView, uProj, uNormal;
out vec3 vWorld;
out vec3 vNrm;
void main(){
  vec4 w = uModel * vec4(aPos, 1.0);
  vWorld = w.xyz;
  vNrm = mat3(uNormal) * aNrm;
  gl_Position = uProj * uView * w;
}`;

const FS_BODY = `#version 300 es
precision highp float;
in vec3 vWorld;
in vec3 vNrm;
uniform vec3 uEye;
uniform float uViz;
out vec4 frag;
${GLSL_FLOW}
void main(){
  vec3 n = normalize(vNrm);
  vec3 p = vWorld + n * 0.04;
  vec3 u = flowWorld(p, uTime);
  float spd = length(u);
  float cp = 1.0 - (spd * spd) / max(uWind * uWind, 0.08);
  vec3 albedo = vec3(0.66, 0.68, 0.7);
  if (uViz > 2.5) {
    albedo = mix(vec3(0.16, 0.42, 0.95), vec3(0.9, 0.22, 0.16), clamp(cp * 0.5 + 0.5, 0.0, 1.0));
  }
  vec3 L = normalize(vec3(0.45, 0.82, 0.32));
  vec3 V = normalize(uEye - vWorld);
  float diff = max(dot(n, L), 0.0);
  float hemi = n.y * 0.5 + 0.5;
  float spec = pow(max(dot(n, normalize(L + V)), 0.0), 48.0) * 0.28;
  vec3 col = albedo * (0.14 + 0.38 * hemi + 0.52 * diff) + spec;
  frag = vec4(col, 1.0);
}`;

const VS_FLOOR = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
uniform mat4 uView, uProj;
out vec3 vWorld;
void main(){
  vWorld = aPos;
  gl_Position = uProj * uView * vec4(aPos, 1.0);
}`;

const FS_FLOOR = `#version 300 es
precision highp float;
in vec3 vWorld;
out vec4 frag;
void main(){
  float g = abs(fract(vWorld.x * 0.5) - 0.5) * abs(fract(vWorld.z * 0.5) - 0.5);
  vec3 col = mix(vec3(0.07, 0.075, 0.08), vec3(0.1, 0.11, 0.115), step(0.18, g));
  float fade = 1.0 - smoothstep(7.0, 11.0, length(vWorld.xz));
  frag = vec4(col, fade);
}`;

export const aeroLab: LabDefinition = {
  id: "aero",
  index: "08",
  title: "Wind Tunnel",
  subtitle: "3D potential · horseshoe · tracers",
  defaultCamera: "orbit",
  pipeline: [
    "3D body VAO",
    "Analytic flow field",
    "Horseshoe vortex",
    "RK2 particle advect",
    "RGBA16F tracer FBO",
    "Cp on the surface",
  ],
  params: [
    {
      key: "body",
      label: "Body",
      min: 0,
      max: 5,
      step: 1,
      default: 2,
      choices: [...BODY3_NAMES],
    },
    { key: "aoa", label: "Angle of attack °", min: -28, max: 28, step: 0.5, default: 8 },
    { key: "size", label: "Scale", min: 0.32, max: 0.95, step: 0.01, default: 0.55 },
    { key: "height", label: "Mount height", min: 0.35, max: 1.35, step: 0.01, default: 0.72 },
    { key: "wind", label: "Freestream U", min: 0.6, max: 2.8, step: 0.05, default: 1.65 },
    { key: "shed", label: "Shedding", min: 0, max: 1.2, step: 0.01, default: 0.7 },
    {
      key: "viz",
      label: "Field",
      min: 0,
      max: 3,
      step: 1,
      default: 0,
      choices: [...VIZ3_NAMES],
    },
    {
      key: "particles",
      label: "Tracers",
      min: 0,
      max: 1,
      step: 1,
      default: 1,
      choices: ["Off", "On"],
    },
  ],
  note: {
    title: "A 3D tunnel, a changeable body",
    body: "Wind is a steady freestream from −X. Tracers are a smoke rake — 14×10 streamlines plus fill — advected with RK2 through an analytic 3D velocity field on the GPU. Sphere uses the exact doublet potential. The wing uses lifting-line theory: a bound vortex along the span and two trailing vortices from the tips (a horseshoe). Circulation Γ = ½ Cl U c with Cl ≈ 2π α, collapsing as the wing stalls. Cube, wedge, plate and diamond block the stream with a no-slip SDF and shed a Kármán street. Swap the body or crank angle of attack and the field updates that frame. Pressure mode paints Cp = 1 − (u/U)² on the metal (Bernoulli). Full 3D Navier–Stokes is a 3D FBO; this is the vortex-method / potential-flow stack wind-tunnel engineers still use to read a flow before they pay for a RANS solve.",
    glsl: "u = U∞ + u_doublet + u_horseshoe(Γ)\nΓ = ½ Cl U c,   Cl ≈ 2π α\nCp = 1 − |u|²/U²\np ← p + ½ (k₁+k₂) Δt     // RK2",
    mapping: "glTexStorage2D(GL_RGBA16F) → particle state\ngl_PointSize + vertex-texture fetch → 10k tracers\nSame as a desktop core-profile particle CFD viewer.",
  },
  create(ctx: GlContext): LabInstance {
    const { gl } = ctx;
    const tri = makeFullscreenTri(gl);
    const field = new ParticleField(gl, tri, PW, PH);
    const ids = particleIds(gl);
    const tracerProg = createProgram(gl, VS_TRACER, FS_TRACER);
    const bodyProg = createProgram(gl, VS_BODY, FS_BODY);
    const floorProg = createProgram(gl, VS_FLOOR, FS_FLOOR);
    const lines = makeLineProgram(gl);

    const sphere = uploadMesh(gl, createIcosphere(1, 3));
    const cube = uploadMesh(gl, createCube(2));
    const wing = uploadMesh(gl, createNacaWing(1.7, 2.6, 0.12, 0.02));
    const wedge = uploadMesh(gl, createWedge(1.15, 1.87));
    const plate = uploadMesh(gl, createCube(1));
    const diamond = uploadMesh(gl, createOctahedron(1.15));
    const floor = uploadMesh(gl, createPlane(18, 1, 0));
    const grid = makeGrid(gl);
    const tunnel = createMesh(gl, createTunnelWires(-5.5, 6.2, 2.6, 2.8).vertices, undefined, gl.LINES);

    const bodies = [sphere, cube, wing, wedge, plate, diamond];
    const model = mat4Identity();
    const tmp = mat4Identity();
    const tmp2 = mat4Identity();
    const normal = mat4Identity();
    let t = 0;
    let cur = pack({});
    let cam: Camera | null = null;

    const buildModel = (p: FlowParams) => {
      mat4FromTranslation(model, [0, p.bodyY, 0]);
      mat4FromRotationZ(tmp, p.aoa);
      mat4Multiply(tmp2, model, tmp);
      if (p.shape === 4) {
        mat4FromScaling(tmp, [p.size * 2.7, p.size * 0.14, p.size * 2.6]);
      } else {
        mat4FromScaling(tmp, [p.size, p.size, p.size]);
      }
      mat4Multiply(model, tmp2, tmp);
    };

    return {
      update(dt, _input, params, camera) {
        cur = pack(params);
        cam = camera;
        t += dt;
        field.step(dt, t, cur);
      },
      draw(w, h) {
        if (!cam) return { draws: 0, tris: 0, instances: 0, gpuMs: null, cpuMs: 0 };
        gl.viewport(0, 0, w, h);
        gl.enable(gl.DEPTH_TEST);
        gl.enable(gl.CULL_FACE);
        gl.depthMask(true);
        gl.disable(gl.BLEND);
        gl.clearColor(0.045, 0.048, 0.052, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.useProgram(floorProg.prog);
        setCameraUniforms(gl, floorProg, cam);
        gl.disable(gl.CULL_FACE);
        drawMesh(gl, floor);
        gl.enable(gl.CULL_FACE);
        gl.disable(gl.BLEND);

        drawGrid(gl, grid, lines, cam);
        gl.useProgram(lines.prog);
        setCameraUniforms(gl, lines, cam);
        gl.disable(gl.CULL_FACE);
        drawMesh(gl, tunnel);
        gl.enable(gl.CULL_FACE);

        buildModel(cur);
        const mesh = bodies[cur.shape] ?? sphere;
        gl.useProgram(bodyProg.prog);
        setCameraUniforms(gl, bodyProg, cam, model);
        gl.uniformMatrix4fv(bodyProg.uniforms.uNormal, false, mat4Normal(normal, model));
        gl.uniform3fv(bodyProg.uniforms.uEye, cam.eye);
        field.bindFlow(bodyProg, cur, t);
        gl.uniform1f(bodyProg.uniforms.uViz, cur.viz);
        drawMesh(gl, mesh);

        let instances = 0;
        if (cur.particles > 0.5) {
          gl.enable(gl.BLEND);
          gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
          gl.depthMask(false);
          gl.useProgram(tracerProg.prog);
          setCameraUniforms(gl, tracerProg, cam);
          field.bindFlow(tracerProg, cur, t);
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, field.tex);
          gl.uniform1i(tracerProg.uniforms.uParticles, 0);
          gl.uniform1f(tracerProg.uniforms.uViz, cur.viz);
          gl.uniform1f(tracerProg.uniforms.uPoint, Math.max(140, h * 0.22));
          drawMesh(gl, ids);
          instances = PW * PH;
          gl.depthMask(true);
          gl.disable(gl.BLEND);
        }

        return {
          draws: 4,
          tris: mesh.count / 3,
          instances,
          gpuMs: null,
          cpuMs: 0,
          extra: {
            tracers: instances,
            body: BODY3_NAMES[cur.shape] ?? "—",
            α: `${((cur.aoa * 180) / Math.PI).toFixed(1)}°`,
            U: cur.wind.toFixed(2),
          },
        };
      },
      dispose() {
        field.dispose();
        disposeCommon(
          gl,
          [tri, ids, sphere, cube, wing, wedge, plate, diamond, floor, grid, tunnel],
          [tracerProg, bodyProg, floorProg, lines],
        );
      },
    };
  },
};
