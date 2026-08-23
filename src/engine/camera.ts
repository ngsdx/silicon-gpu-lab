import {
  clamp,
  mat4LookAt,
  mat4Perspective,
  vec3,
  vec3Mad,
  vec3Normalize,
  vec3Set,
  type Mat4,
  type Vec3,
} from "./math";
import type { InputState } from "./input";

const WORLD_UP = vec3(0, 1, 0);

export type CameraMode = "orbit" | "fly";

export class Camera {
  mode: CameraMode = "orbit";
  target: Vec3 = vec3(0, 0.6, 0);
  eye: Vec3 = vec3(0, 1.6, 4.2);
  yaw = 0.35;
  pitch = 0.38;
  distance = 6.2;
  fov = 55 * (Math.PI / 180);
  near = 0.08;
  far = 120;
  moveSpeed = 4.2;
  lookSens = 0.005;
  orbitSens = 0.005;
  view: Mat4 = new Float32Array(16);
  proj: Mat4 = new Float32Array(16);
  viewProj: Mat4 = new Float32Array(16);
  forward: Vec3 = vec3(0, 0, -1);
  right: Vec3 = vec3(1, 0, 0);
  speed = 0;

  resize(aspect: number) {
    mat4Perspective(this.proj, this.fov, Math.max(aspect, 0.05), this.near, this.far);
  }

  applyInput(input: InputState, dt: number) {
    if (this.mode === "orbit") this.orbit(input, dt);
    else this.fly(input, dt);
    this.rebuild();
  }

  private orbit(input: InputState, _dt: number) {
    if (input.pointerDown && input.pointers === 1) {
      this.yaw -= input.dx * this.orbitSens;
      this.pitch = clamp(this.pitch + input.dy * this.orbitSens, -1.2, 1.45);
    }
    if (input.pointers >= 2 && input.pinchDelta) {
      this.distance = clamp(this.distance * (1 - input.pinchDelta * 0.01), 1.2, 28);
    }
    this.distance = clamp(this.distance * (1 - input.wheel * 0.001), 1.2, 28);
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    this.eye[0] = this.target[0]! + sy * cp * this.distance;
    this.eye[1] = this.target[1]! + sp * this.distance;
    this.eye[2] = this.target[2]! + cy * cp * this.distance;
    this.speed = 0;
  }

  private fly(input: InputState, dt: number) {
    if (input.pointerLocked || input.pointerDown) {
      this.yaw -= input.dx * this.lookSens;
      this.pitch = clamp(this.pitch - input.dy * this.lookSens, -1.4, 1.4);
    }
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    // FPS on-foot: yaw 0 faces -Z; A = -right, D = +right, movement uses yaw only.
    vec3Set(this.forward, -sy, 0, -cy);
    vec3Set(this.right, cy, 0, -sy);
    vec3Normalize(this.forward, this.forward);
    vec3Normalize(this.right, this.right);

    const sprint = input.actions.sprint ? 2.4 : 1;
    const v = this.moveSpeed * sprint;
    let mx = 0, mz = 0;
    mz += input.actions.moveY;
    mx += input.actions.moveX;
    const mag = Math.hypot(mx, mz) || 1;
    mx /= mag;
    mz /= mag;
    vec3Mad(this.eye, this.eye, this.forward, mz * v * dt);
    vec3Mad(this.eye, this.eye, this.right, mx * v * dt);
    this.eye[1] += input.actions.lift * v * dt;
    this.speed = Math.hypot(mx, mz, input.actions.lift) * v;
    vec3Set(this.target, this.eye[0]! + this.forward[0]!, this.eye[1]!, this.eye[2]! + this.forward[2]!);
  }

  rebuild() {
    if (this.mode === "fly") {
      const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
      const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
      const look = vec3(-sy * cp, sp, -cy * cp);
      vec3Set(this.target, this.eye[0]! + look[0]!, this.eye[1]! + look[1]!, this.eye[2]! + look[2]!);
    }
    mat4LookAt(this.view, this.eye, this.target, WORLD_UP);
  }
}

export function bindControlsTest(cam: Camera) {
  if (typeof window === "undefined") return;
  window.__controlsTest = {
    getYaw: () => cam.yaw,
    getSpeed: () => cam.speed,
    setSteer: () => undefined,
    setKeys: (codes: string[]) => {
      // Injected by QA; Viewport writes the actual key set.
      const w = window as Window & { __qaKeys?: Set<string> };
      w.__qaKeys = new Set(codes);
    },
  };
}

declare global {
  interface Window {
    __controlsTest?: {
      getYaw: () => number;
      getSpeed: () => number;
      setSteer?: (v: number) => void;
      setKeys?: (codes: string[]) => void;
    };
    __qaKeys?: Set<string>;
  }
}
