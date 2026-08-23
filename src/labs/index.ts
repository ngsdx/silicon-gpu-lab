import type { LabDefinition } from "@/engine/lab-types";
import { helloTriangleLab } from "./hello-triangle";
import { transformsLab } from "./transforms";
import { phongLab } from "./phong";
import { samplingLab } from "./sampling";
import { shadowsLab } from "./shadows";
import { instancingLab } from "./instancing";
import { gbufferLab } from "./gbuffer";
import { softwareLab } from "./software";

export const LABS: LabDefinition[] = [
  helloTriangleLab,
  transformsLab,
  phongLab,
  samplingLab,
  shadowsLab,
  instancingLab,
  gbufferLab,
  softwareLab,
];

export function labById(id: string): LabDefinition {
  return LABS.find((l) => l.id === id) ?? LABS[0]!;
}
