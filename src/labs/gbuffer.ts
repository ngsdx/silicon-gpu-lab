import { drawMesh } from "@/engine/buffer";
import { createProgram } from "@/engine/shader";
import { disposeCommon, makeFullscreenTri, uploadMesh, setCameraUniforms } from "@/engine/common-gl";
import { createCube, createIcosphere, createPlane, createTorus } from "@/engine/mesh";
import { bindFramebuffer, createFramebuffer, destroyFramebuffer } from "@/engine/framebuffer";
import { mat4FromRotationY, mat4FromTranslation, mat4Identity, mat4Multiply, mat4Normal } from "@/engine/math";
import type { Camera } from "@/engine/camera";
import type { LabDefinition, LabInstance } from "@/engine/lab-types";
import type { GlContext } from "@/engine/gpu";

const VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec2 aUv;
uniform mat4 uModel, uView, uProj, uNormal;
out vec3 vNrm; out vec3 vView; out vec2 vUv;
void main(){
  vec4 view = uView * uModel * vec4(aPos,1.0);
  vView = view.xyz;
  vNrm = mat3(uView) * mat3(uNormal) * aNrm;
  vUv = aUv;
  gl_Position = uProj * view;
}`;

const FS = `#version 300 es
precision highp float;
in vec3 vNrm; in vec3 vView; in vec2 vUv;
uniform vec3 uAlbedo;
layout(location=0) out vec4 oAlbedo;
layout(location=1) out vec4 oNormal;
layout(location=2) out vec4 oDepth;
void main(){
  oAlbedo = vec4(uAlbedo, 1.0);
  vec3 n = normalize(vNrm) * 0.5 + 0.5;
  oNormal = vec4(n, 1.0);
  float z = length(vView);
  oDepth = vec4(vec3(exp(-z * 0.12)), 1.0);
}`;

const VS_BLIT = `#version 300 es
layout(location=0) in vec2 aPos; out vec2 vUv;
void main(){ vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos,0.0,1.0); }`;
const FS_BLIT = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uA, uN, uD;
uniform float uMode;
out vec4 frag;
void main(){
  if (uMode < 0.5) frag = texture(uA, vUv);
  else if (uMode < 1.5) frag = texture(uN, vUv);
  else if (uMode < 2.5) frag = texture(uD, vUv);
  else {
    vec2 uv = vUv;
    if (uv.x < 0.333) frag = texture(uA, vec2(uv.x*3.0, uv.y));
    else if (uv.x < 0.666) frag = texture(uN, vec2((uv.x-0.333)*3.0, uv.y));
    else frag = texture(uD, vec2((uv.x-0.666)*3.0, uv.y));
  }
}`;

export const gbufferLab: LabDefinition = {
  id: "gbuffer",
  index: "06",
  title: "G-Buffer",
  subtitle: "MRT · deferred geometry",
  pipeline: ["Geometry pass", "MRT", "Albedo", "VS-normal", "Linear depth"],
  params: [
    { key: "mode", label: "View (0 alb, 1 nrm, 2 z, 3 split)", min: 0, max: 3, step: 1, default: 3 },
    { key: "spin", label: "Spin", min: 0, max: 1.5, step: 0.01, default: 0.3 },
  ],
  note: {
    title: "Multiple render targets, one geometry pass",
    body: "Deferred shading splits the frame into a fat G-buffer: albedo, view-space normals, depth. One geometry pass writes all three via MRT (glDrawBuffers). Lighting then becomes a fullscreen pass that reads those textures — lights no longer scale with scene triangle count, they scale with pixels × lights. This lab visualizes the buffers themselves, the thing a GPU debugger would show you.",
    glsl: "layout(location=0) out vec4 oAlbedo;\nlayout(location=1) out vec4 oNormal;\nlayout(location=2) out vec4 oDepth;",
    mapping: "glDrawBuffers({COLOR0, COLOR1, COLOR2})\nglFramebufferTexture2D per attachment\nEXT_draw_buffers is core in ES 3.00 / GL 3.0",
  },
  create(ctx: GlContext): LabInstance {
    const { gl } = ctx;
    let fbW = 0, fbH = 0;
    let fb = createFramebuffer(gl, 4, 4, [gl.RGBA8, gl.RGBA8, gl.RGBA8], "renderbuffer");
    const sphere = uploadMesh(gl, createIcosphere(0.8, 3));
    const torus = uploadMesh(gl, createTorus(0.75, 0.28, 40, 20));
    const cube = uploadMesh(gl, createCube(0.9));
    const plane = uploadMesh(gl, createPlane(14, 1, 0));
    const geo = createProgram(gl, VS, FS);
    const blit = createProgram(gl, VS_BLIT, FS_BLIT);
    const tri = makeFullscreenTri(gl);
    const model = mat4Identity();
    const normal = mat4Identity();
    const tmp = mat4Identity();
    const tmp2 = mat4Identity();
    let cam: Camera | null = null;
    let t = 0, mode = 3, spin = 0.3;

    const ensure = (w: number, h: number) => {
      if (w === fbW && h === fbH) return;
      destroyFramebuffer(gl, fb);
      fb = createFramebuffer(gl, w, h, [gl.RGBA8, gl.RGBA8, gl.RGBA8], "renderbuffer");
      fbW = w;
      fbH = h;
    };

    const drawObj = (mesh: typeof sphere, pos: number[], albedo: number[], yaw: number) => {
      if (!cam) return;
      mat4FromTranslation(tmp, pos);
      mat4FromRotationY(tmp2, yaw);
      mat4Multiply(model, tmp, tmp2);
      setCameraUniforms(gl, geo, cam, model);
      gl.uniformMatrix4fv(geo.uniforms.uNormal, false, mat4Normal(normal, model));
      gl.uniform3fv(geo.uniforms.uAlbedo, albedo);
      drawMesh(gl, mesh);
    };

    return {
      update(dt, _i, params, camera) {
        t += dt;
        cam = camera;
        mode = params.mode ?? 3;
        spin = params.spin ?? 0.3;
      },
      draw(w, h) {
        if (!cam) return { draws: 0, tris: 0, instances: 0, gpuMs: null, cpuMs: 0 };
        ensure(w, h);
        bindFramebuffer(gl, fb);
        gl.clearColor(0.02, 0.022, 0.025, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.useProgram(geo.prog);
        drawObj(plane, [0, 0, 0], [0.16, 0.17, 0.18], 0);
        drawObj(sphere, [0, 0.9, 0], [0.82, 0.3, 0.26], t * spin);
        drawObj(torus, [-2.0, 0.7, 0.3], [0.7, 0.74, 0.68], t * spin * 1.4);
        drawObj(cube, [2.05, 0.45, -0.2], [0.38, 0.56, 0.74], t * spin * 0.6);

        bindFramebuffer(gl, null, w, h);
        gl.disable(gl.DEPTH_TEST);
        gl.useProgram(blit.prog);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, fb.colors[0]!);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, fb.colors[1]!);
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, fb.colors[2]!);
        gl.uniform1i(blit.uniforms.uA, 0);
        gl.uniform1i(blit.uniforms.uN, 1);
        gl.uniform1i(blit.uniforms.uD, 2);
        gl.uniform1f(blit.uniforms.uMode, mode);
        drawMesh(gl, tri);
        gl.enable(gl.DEPTH_TEST);
        return {
          draws: 5,
          tris: sphere.count / 3 + torus.count / 3 + 14,
          instances: 4,
          gpuMs: null,
          cpuMs: 0,
          extra: { MRT: "3× RGBA8", mode: ["albedo", "normal", "depth", "split"][mode | 0] ?? "split" },
        };
      },
      dispose() {
        destroyFramebuffer(gl, fb);
        disposeCommon(gl, [sphere, torus, cube, plane, tri], [geo, blit]);
      },
    };
  },
};
