import { drawMesh } from "@/engine/buffer";
import { createProgram } from "@/engine/shader";
import {
  disposeCommon,
  drawGrid,
  makeAxes,
  makeGrid,
  makeLineProgram,
  setCameraUniforms,
  uploadMesh,
  VS_LIT,
  FS_UNLIT,
} from "@/engine/common-gl";
import { createCube, createTorus } from "@/engine/mesh";
import {
  mat4FromRotationY,
  mat4FromScaling,
  mat4FromTranslation,
  mat4Identity,
  mat4Multiply,
  mat4Normal,
} from "@/engine/math";
import type { Camera } from "@/engine/camera";
import type { LabDefinition, LabInstance } from "@/engine/lab-types";
import type { GlContext } from "@/engine/gpu";

export const transformsLab: LabDefinition = {
  id: "transforms",
  index: "01",
  title: "Coordinate Spaces",
  subtitle: "Model · View · Projection",
  pipeline: ["Model", "View", "Projection", "Clip", "NDC", "Viewport"],
  params: [
    { key: "spin", label: "Orbit speed", min: 0, max: 2, step: 0.01, default: 0.45 },
    { key: "fov", label: "Vertical FOV", min: 30, max: 90, step: 1, default: 55 },
  ],
  note: {
    title: "The matrix stack, built by hand",
    body: "clip = P · V · M · vertex. M puts the mesh in the world, V is the inverse camera, P is the frustum. These four matrices are multiplied here in JavaScript — column-major, matching glUniformMatrix4fv. Drag to orbit. Switch to Fly in the top bar: WASD is FPS (A left, D right), Q/E up-down, Shift sprint.",
    glsl: "gl_Position = uProj * uView * uModel * vec4(aPos, 1.0);",
    mapping: "gluLookAt / glm::lookAt → mat4LookAt()\ngluPerspective → mat4Perspective()\nglUniformMatrix4fv → uniformMatrix4fv",
  },
  create(ctx: GlContext): LabInstance {
    const { gl } = ctx;
    const cube = uploadMesh(gl, createCube(1.05));
    const torus = uploadMesh(gl, createTorus(0.82, 0.2, 40, 16));
    const grid = makeGrid(gl);
    const axes = makeAxes(gl);
    const lit = createProgram(gl, VS_LIT, FS_UNLIT);
    const lines = makeLineProgram(gl);
    const model = mat4Identity();
    const normal = mat4Identity();
    const tmp = mat4Identity();
    const tmp2 = mat4Identity();
    let t = 0;
    let spin = 0.45;
    let fov = 55;
    let cam: Camera | null = null;

    const drawObject = (mesh: typeof cube, color: number[], tris: number) => {
      gl.useProgram(lit.prog);
      if (cam) setCameraUniforms(gl, lit, cam, model);
      gl.uniformMatrix4fv(lit.uniforms.uNormal, false, mat4Normal(normal, model));
      gl.uniform3fv(lit.uniforms.uColor, color);
      drawMesh(gl, mesh);
      return tris;
    };

    return {
      update(dt, _i, params, camera) {
        t += dt;
        spin = params.spin ?? 0.45;
        fov = params.fov ?? 55;
        camera.fov = (fov * Math.PI) / 180;
        cam = camera;
      },
      draw() {
        if (!cam) return { draws: 0, tris: 0, instances: 0, gpuMs: null, cpuMs: 0 };
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        let draws = 0;
        let tris = 0;
        drawGrid(gl, grid, lines, cam);
        draws++;
        gl.useProgram(lines.prog);
        setCameraUniforms(gl, lines, cam);
        gl.disable(gl.DEPTH_TEST);
        drawMesh(gl, axes);
        gl.enable(gl.DEPTH_TEST);
        draws++;

        mat4FromRotationY(tmp, t * spin);
        mat4FromScaling(tmp2, [1, 1, 1]);
        mat4Multiply(model, tmp, tmp2);
        tris += drawObject(cube, [0.82, 0.84, 0.8], 12);
        draws++;

        mat4FromTranslation(tmp, [Math.cos(t * spin) * 2.2, 0.35, Math.sin(t * spin) * 2.2]);
        mat4FromRotationY(tmp2, t * spin * 1.7);
        mat4Multiply(model, tmp, tmp2);
        tris += drawObject(torus, [0.55, 0.78, 0.7], torus.count / 3);
        draws++;

        return {
          draws,
          tris,
          instances: 2,
          gpuMs: null,
          cpuMs: 0,
          extra: { "M[0]": model[0]!.toFixed(2), FOV: `${fov}°` },
        };
      },
      dispose() {
        disposeCommon(gl, [cube, torus, grid, axes], [lit, lines]);
      },
    };
  },
};
