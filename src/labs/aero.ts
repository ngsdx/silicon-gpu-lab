import { createRawMesh, type GpuMesh } from "@/engine/buffer";
import { makeFullscreenTri, disposeCommon } from "@/engine/common-gl";
import { BODY_NAMES, Fluid2D, VIZ_NAMES, type FluidParams } from "@/engine/fluid2d";
import type { LabDefinition, LabInstance } from "@/engine/lab-types";
import type { GlContext } from "@/engine/gpu";

function particleMesh(gl: WebGL2RenderingContext, w: number, h: number): GpuMesh {
  const n = w * h;
  const uv = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    uv[i * 2] = ((i % w) + 0.5) / w;
    uv[i * 2 + 1] = (Math.floor(i / w) + 0.5) / h;
  }
  return createRawMesh(gl, uv, 2, gl.POINTS);
}

function pack(params: Record<string, number>): FluidParams {
  return {
    shape: Math.round(params.body ?? 0),
    aoa: ((params.aoa ?? 0) * Math.PI) / 180,
    size: params.size ?? 0.11,
    bodyY: params.height ?? 0.5,
    wind: params.wind ?? 0.2,
    visc: params.visc ?? 0.00012,
    confine: params.confine ?? 0.4,
    viz: Math.round(params.viz ?? 0),
    particles: params.particles ?? 1,
  };
}

export const aeroLab: LabDefinition = {
  id: "aero",
  index: "08",
  title: "Wind Tunnel",
  subtitle: "Navier–Stokes · no-slip · Kármán",
  hideCamera: true,
  pipeline: [
    "RGBA16F velocity FBO",
    "Semi-Lagrangian advect",
    "Vorticity confinement",
    "Jacobi pressure Poisson",
    "Project (make ∇·u = 0)",
    "No-slip body SDF",
    "Dye + particle tracers",
  ],
  params: [
    {
      key: "body",
      label: "Body",
      min: 0,
      max: 5,
      step: 1,
      default: 0,
      choices: [...BODY_NAMES],
    },
    { key: "aoa", label: "Angle of attack °", min: -28, max: 28, step: 0.5, default: 8 },
    { key: "size", label: "Chord / diameter", min: 0.055, max: 0.17, step: 0.001, default: 0.11 },
    { key: "height", label: "Vertical station", min: 0.28, max: 0.72, step: 0.01, default: 0.5 },
    { key: "wind", label: "Freestream U", min: 0.08, max: 0.38, step: 0.005, default: 0.2 },
    { key: "visc", label: "Viscosity ν", min: 0.00002, max: 0.0008, step: 0.00001, default: 0.0001 },
    { key: "confine", label: "Vorticity ε", min: 0, max: 0.7, step: 0.01, default: 0.4 },
    {
      key: "viz",
      label: "Field",
      min: 0,
      max: 3,
      step: 1,
      default: 0,
      choices: [...VIZ_NAMES],
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
    title: "A one-way wind, a changeable body",
    body: "The tunnel solves the 2D incompressible Navier–Stokes equations on the GPU — the same projection method used in every stable-fluids solver. Freestream is Dirichlet on the left (steady, one direction). The body is a signed-distance field restamped every frame, so swapping Cylinder → Airfoil → Plate is instant: no-slip walls move, the wake has to renegotiate. Cylinder at this Reynolds number sheds a Kármán street. The airfoil at modest α keeps attached flow; crank angle of attack and the upper surface separates. Plate broadside is a bluff-body lesson. Dye is a conserved tracer; vorticity is ∇×u; pressure is the Lagrange multiplier that enforced incompressibility this step.",
    glsl: "u ← advect(u)\nu ← u + ε ω × ∇|ω|     // confinement\n∇²p = ∇·u              // Jacobi on an FBO\nu ← u − ∇p              // Helmholtz-Hodge\nu = 0  on the body     // no-slip",
    mapping: "glTexStorage2D(GL_RGBA16F) → texStorage2D(RGBA16F)\nglFramebufferTexture2D → framebufferTexture2D\nPing-pong two FBOs per field, same as a compute-shader CFD code.\nEXT_color_buffer_float is the ES equivalent of rendering to a 16-bit float texture.",
  },
  create(ctx: GlContext): LabInstance {
    const { gl } = ctx;
    const tri = makeFullscreenTri(gl);
    const W = 256;
    const H = 128;
    const fluid = new Fluid2D(gl, tri, W, H);
    const points = particleMesh(gl, 128, 48);
    let t = 0;
    let cur: FluidParams = pack({});
    fluid.reset(cur.wind);

    return {
      update(dt, _input, params) {
        const next = pack(params);
        if (Math.abs(next.wind - cur.wind) > 0.1) fluid.reset(next.wind);
        cur = next;
        t += dt;
        fluid.step(dt, t, cur);
      },
      draw(w, h) {
        fluid.display(w, h, cur, points);
        const Re = (cur.wind * (2 * cur.size)) / Math.max(cur.visc, 1e-8);
        return {
          draws: 1,
          tris: 1,
          instances: 128 * 48,
          gpuMs: null,
          cpuMs: 0,
          extra: {
            grid: `${W}×${H}`,
            Re: Re.toFixed(0),
            body: BODY_NAMES[cur.shape] ?? "—",
            FBO: "RGBA16F",
          },
        };
      },
      dispose() {
        fluid.dispose();
        disposeCommon(gl, [tri, points], []);
      },
    };
  },
};
