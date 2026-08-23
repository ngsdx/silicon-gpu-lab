import type { Camera } from "./camera";
import type { GlContext } from "./gpu";
import type { InputState } from "./input";

export type LabParam = {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
  choices?: string[];
};

export type LabStats = {
  draws: number;
  tris: number;
  instances: number;
  gpuMs: number | null;
  cpuMs: number;
  extra?: Record<string, string | number>;
};

export type LabNote = {
  title: string;
  body: string;
  glsl?: string;
  mapping?: string;
};

export type LabInstance = {
  update(dt: number, input: InputState, params: Record<string, number>, camera: Camera): void;
  draw(width: number, height: number): LabStats;
  dispose(): void;
  getShader?: () => { vs: string; fs: string };
  setShader?: (vs: string, fs: string) => string | null;
};

export type LabDefinition = {
  id: string;
  index: string;
  title: string;
  subtitle: string;
  pipeline: string[];
  params: LabParam[];
  note: LabNote;
  supportsShaderEdit?: boolean;
  defaultCamera?: "orbit" | "fly";
  hideCamera?: boolean;
  fileUpload?: { accept: string; hint: string };
  create(ctx: GlContext): LabInstance;
};
