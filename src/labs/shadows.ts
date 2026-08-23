import { drawMesh } from "@/engine/buffer";
import { createProgram } from "@/engine/shader";
import { disposeCommon, makeFullscreenTri, uploadMesh, setCameraUniforms } from "@/engine/common-gl";
import { createCube, createIcosphere, createPlane, createTorus } from "@/engine/mesh";
import { bindFramebuffer, createFramebuffer, destroyFramebuffer, type Framebuffer } from "@/engine/framebuffer";
import {
  mat4FromRotationY,
  mat4FromTranslation,
  mat4Identity,
  mat4LookAt,
  mat4Multiply,
  mat4Normal,
  mat4Ortho,
} from "@/engine/math";
import type { Camera } from "@/engine/camera";
import type { LabDefinition, LabInstance } from "@/engine/lab-types";
import type { GlContext } from "@/engine/gpu";

const VS_DEPTH = `#version 300 es
layout(location=0) in vec3 aPos;
uniform mat4 uLightVP, uModel;
void main(){ gl_Position = uLightVP * uModel * vec4(aPos,1.0); }`;
const FS_DEPTH = `#version 300 es
precision highp float;
void main(){}`;

const VS_SCENE = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
uniform mat4 uModel, uView, uProj, uNormal, uLightVP;
out vec3 vWorld; out vec3 vNrm; out vec4 vShadow;
void main(){
  vec4 w = uModel * vec4(aPos,1.0);
  vWorld = w.xyz;
  vNrm = mat3(uNormal) * aNrm;
  vShadow = uLightVP * w;
  gl_Position = uProj * uView * w;
}`;

const FS_SCENE = `#version 300 es
precision highp float;
precision highp sampler2DShadow;
in vec3 vWorld; in vec3 vNrm; in vec4 vShadow;
uniform sampler2DShadow uShadow;
uniform vec3 uLightDir, uAlbedo, uEye;
uniform float uBias, uPcf;
out vec4 frag;
float shadowAt(vec3 uvz){
  if (uvz.z > 1.0 || uvz.x<0.0||uvz.x>1.0||uvz.y<0.0||uvz.y>1.0) return 1.0;
  if (uPcf < 0.5) return texture(uShadow, uvz);
  float s = 0.0;
  vec2 texel = vec2(1.0/1024.0);
  for (int y=-1;y<=1;y++)
    for (int x=-1;x<=1;x++)
      s += texture(uShadow, uvz + vec3(vec2(x,y)*texel, 0.0));
  return s / 9.0;
}
void main(){
  vec3 n = normalize(vNrm);
  vec3 L = normalize(-uLightDir);
  float ndl = max(dot(n, L), 0.0);
  vec3 proj = vShadow.xyz / vShadow.w;
  vec3 uvz = vec3(proj.xy * 0.5 + 0.5, proj.z * 0.5 + 0.5 - uBias);
  float vis = shadowAt(uvz);
  vec3 col = uAlbedo * (0.07 + 0.93 * ndl * vis);
  float spec = pow(max(dot(n, normalize(L + normalize(uEye-vWorld))), 0.0), 48.0) * vis * 0.35;
  frag = vec4(col + spec, 1.0);
}`;

const FS_DEPTH_VIZ = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTex;
out vec4 frag;
void main(){
  float z = texture(uTex, vUv).r;
  z = z * z * z;
  frag = vec4(vec3(z), 1.0);
}`;

export const shadowsLab: LabDefinition = {
  id: "shadows",
  index: "04",
  title: "Shadow Maps",
  subtitle: "FBO · depth compare · PCF",
  pipeline: ["Light pass", "Depth FBO", "Projective tex", "PCF", "Shade"],
  params: [
    { key: "bias", label: "Depth bias", min: 0, max: 0.02, step: 0.0005, default: 0.004 },
    { key: "pcf", label: "PCF 3×3", min: 0, max: 1, step: 1, default: 1 },
    { key: "spin", label: "Light yaw", min: 0, max: 6.28, step: 0.01, default: 0.9 },
    { key: "debug", label: "Show depth", min: 0, max: 1, step: 1, default: 1 },
  ],
  note: {
    title: "A second camera, from the light",
    body: "Shadow mapping is a depth prepass rendered from the light into a framebuffer, then compared per fragment via a hardware sampler2DShadow (PCF is a 3×3 tap of that compare). Acne is a precision fight — bias pushes the test back along the light ray. Peter-panning is the opposite failure. The depth inset is the actual 1024² DEPTH_COMPONENT32F texture the scene is sampling.",
    glsl: "uniform sampler2DShadow uShadow;\nfloat vis = texture(uShadow, vec3(uv, depth));",
    mapping: "glFramebufferTexture2D(GL_DEPTH_ATTACHMENT)\nglTexParameteri(GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE)\nglDrawBuffer(GL_NONE) on a depth-only FBO",
  },
  create(ctx: GlContext): LabInstance {
    const { gl } = ctx;
    const MAP = 1024;
    const shadowFb: Framebuffer = createFramebuffer(gl, MAP, MAP, [], "texture");
    const sphere = uploadMesh(gl, createIcosphere(0.7, 2));
    const torus = uploadMesh(gl, createTorus(0.55, 0.2, 32, 16));
    const cube = uploadMesh(gl, createCube(0.9));
    const plane = uploadMesh(gl, createPlane(16, 1, 0));
    const depthProg = createProgram(gl, VS_DEPTH, FS_DEPTH);
    const sceneProg = createProgram(gl, VS_SCENE, FS_SCENE);
    const vizProg = createProgram(gl, `#version 300 es
layout(location=0) in vec2 aPos; out vec2 vUv;
void main(){ vUv = aPos*0.5+0.5; gl_Position=vec4(aPos,0.0,1.0); }`, FS_DEPTH_VIZ);
    const fsTri = makeFullscreenTri(gl);
    const lightVP = mat4Identity();
    const lightView = mat4Identity();
    const lightProj = mat4Identity();
    const model = mat4Identity();
    const normal = mat4Identity();
    const tmp = mat4Identity();
    let cam: Camera | null = null;
    let t = 0, bias = 0.004, pcf = 1, yaw = 0.9, debug = 1;

    const objects = [
      { mesh: sphere, pos: [0, 0.95, 0] as const, albedo: [0.78, 0.32, 0.28], spin: 0.4 },
      { mesh: torus, pos: [-1.8, 0.55, 0.6] as const, albedo: [0.7, 0.72, 0.68], spin: 0.7 },
      { mesh: cube, pos: [1.9, 0.45, -0.4] as const, albedo: [0.4, 0.58, 0.72], spin: 0.25 },
      { mesh: plane, pos: [0, 0, 0] as const, albedo: [0.22, 0.23, 0.24], spin: 0 },
    ];

    const place = (o: (typeof objects)[number]) => {
      mat4FromTranslation(tmp, o.pos);
      mat4FromRotationY(model, t * o.spin);
      mat4Multiply(model, tmp, model);
    };

    return {
      update(dt, _i, params, camera) {
        t += dt;
        cam = camera;
        bias = params.bias ?? 0.004;
        pcf = params.pcf ?? 1;
        yaw = params.spin ?? 0.9;
        debug = params.debug ?? 1;
      },
      draw(w, h) {
        if (!cam) return { draws: 0, tris: 0, instances: 0, gpuMs: null, cpuMs: 0 };
        const lx = Math.cos(yaw) * 8;
        const lz = Math.sin(yaw) * 8;
        mat4LookAt(lightView, [lx, 9, lz], [0, 0, 0], [0, 1, 0]);
        mat4Ortho(lightProj, -8, 8, -8, 8, 0.5, 28);
        mat4Multiply(lightVP, lightProj, lightView);

        bindFramebuffer(gl, shadowFb);
        gl.clear(gl.DEPTH_BUFFER_BIT);
        gl.colorMask(false, false, false, false);
        gl.cullFace(gl.FRONT);
        gl.useProgram(depthProg.prog);
        gl.uniformMatrix4fv(depthProg.uniforms.uLightVP, false, lightVP);
        let draws = 0, tris = 0;
        for (const o of objects) {
          place(o);
          gl.uniformMatrix4fv(depthProg.uniforms.uModel, false, model);
          drawMesh(gl, o.mesh);
          draws++;
          tris += o.mesh.count / 3;
        }
        gl.cullFace(gl.BACK);
        gl.colorMask(true, true, true, true);

        bindFramebuffer(gl, null, w, h);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.useProgram(sceneProg.prog);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, shadowFb.depth as WebGLTexture);
        gl.uniform1i(sceneProg.uniforms.uShadow, 0);
        gl.uniform3f(sceneProg.uniforms.uLightDir, -lx, -9, -lz);
        gl.uniform3fv(sceneProg.uniforms.uEye, cam.eye);
        gl.uniform1f(sceneProg.uniforms.uBias, bias);
        gl.uniform1f(sceneProg.uniforms.uPcf, pcf);
        gl.uniformMatrix4fv(sceneProg.uniforms.uLightVP, false, lightVP);
        for (const o of objects) {
          place(o);
          setCameraUniforms(gl, sceneProg, cam, model);
          gl.uniformMatrix4fv(sceneProg.uniforms.uNormal, false, mat4Normal(normal, model));
          gl.uniform3fv(sceneProg.uniforms.uAlbedo, o.albedo);
          drawMesh(gl, o.mesh);
          draws++;
        }

        if (debug > 0.5) {
          const inset = Math.floor(Math.min(w, h) * 0.22);
          gl.viewport(12, 12, inset, inset);
          const depthTex = shadowFb.depth as WebGLTexture;
          gl.bindTexture(gl.TEXTURE_2D, depthTex);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.NONE);
          gl.disable(gl.DEPTH_TEST);
          gl.useProgram(vizProg.prog);
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, depthTex);
          gl.uniform1i(vizProg.uniforms.uTex, 0);
          drawMesh(gl, fsTri);
          gl.enable(gl.DEPTH_TEST);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL);
          gl.viewport(0, 0, w, h);
          draws++;
        }

        return { draws, tris, instances: 4, gpuMs: null, cpuMs: 0, extra: { map: `${MAP}²`, pcf: pcf ? "3×3" : "1 tap" } };
      },
      dispose() {
        destroyFramebuffer(gl, shadowFb);
        disposeCommon(gl, [sphere, torus, cube, plane, fsTri], [depthProg, sceneProg, vizProg]);
      },
    };
  },
};

