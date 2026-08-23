export type InputActions = {
  moveX: number;
  moveY: number;
  lift: number;
  sprint: boolean;
};

export type InputState = {
  keys: Set<string>;
  actions: InputActions;
  dx: number;
  dy: number;
  wheel: number;
  pointerDown: boolean;
  pointerLocked: boolean;
  pointers: number;
  pinchDelta: number;
};

const GAME_CODES = new Set([
  "KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE", "ShiftLeft", "ShiftRight",
  "Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
]);

export function createInput(target: HTMLElement): {
  state: InputState;
  sample: () => void;
  dispose: () => void;
  requestFlyLock: () => void;
} {
  const keys = new Set<string>();
  const state: InputState = {
    keys,
    actions: { moveX: 0, moveY: 0, lift: 0, sprint: false },
    dx: 0,
    dy: 0,
    wheel: 0,
    pointerDown: false,
    pointerLocked: false,
    pointers: 0,
    pinchDelta: 0,
  };

  const pts = new Map<number, { x: number; y: number }>();
  let lastPinch = 0;
  let accX = 0, accY = 0, accW = 0, accPinch = 0;

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    keys.add(e.code);
    if (GAME_CODES.has(e.code) && !isTyping(e.target)) e.preventDefault();
  };
  const onKeyUp = (e: KeyboardEvent) => keys.delete(e.code);
  const clearKeys = () => keys.clear();

  const onPointerDown = (e: PointerEvent) => {
    if (e.button !== 0 && e.pointerType === "mouse") return;
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    state.pointers = pts.size;
    state.pointerDown = true;
    target.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent) => {
    if (state.pointerLocked) {
      accX += e.movementX;
      accY += e.movementY;
      return;
    }
    const prev = pts.get(e.pointerId);
    if (!prev) return;
    if (pts.size === 1) {
      accX += e.clientX - prev.x;
      accY += e.clientY - prev.y;
    }
    prev.x = e.clientX;
    prev.y = e.clientY;
    if (pts.size >= 2) {
      const arr = [...pts.values()];
      const d = Math.hypot(arr[0]!.x - arr[1]!.x, arr[0]!.y - arr[1]!.y);
      if (lastPinch) accPinch += d - lastPinch;
      lastPinch = d;
    }
  };
  const onPointerUp = (e: PointerEvent) => {
    pts.delete(e.pointerId);
    state.pointers = pts.size;
    if (pts.size === 0) {
      state.pointerDown = false;
      lastPinch = 0;
    }
  };
  const onWheel = (e: WheelEvent) => {
    accW += -e.deltaY;
    e.preventDefault();
  };
  const onLockChange = () => {
    state.pointerLocked = document.pointerLockElement === target;
  };

  target.addEventListener("pointerdown", onPointerDown);
  target.addEventListener("pointermove", onPointerMove);
  target.addEventListener("pointerup", onPointerUp);
  target.addEventListener("pointercancel", onPointerUp);
  target.addEventListener("wheel", onWheel, { passive: false });
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
  window.addEventListener("blur", clearKeys);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) clearKeys();
  });
  document.addEventListener("pointerlockchange", onLockChange);

  const sample = () => {
    if (typeof window !== "undefined" && window.__qaKeys) {
      keys.clear();
      for (const c of window.__qaKeys) keys.add(c);
    }
    state.dx = accX;
    state.dy = accY;
    state.wheel = accW;
    state.pinchDelta = accPinch;
    accX = accY = accW = accPinch = 0;

    let mx = 0, my = 0, lift = 0;
    if (keys.has("KeyD") || keys.has("ArrowRight")) mx += 1;
    if (keys.has("KeyA") || keys.has("ArrowLeft")) mx -= 1;
    if (keys.has("KeyW") || keys.has("ArrowUp")) my += 1;
    if (keys.has("KeyS") || keys.has("ArrowDown")) my -= 1;
    if (keys.has("KeyE") || keys.has("Space")) lift += 1;
    if (keys.has("KeyQ")) lift -= 1;
    const pad = samplePad();
    mx += pad.x;
    my += pad.y;
    state.actions.moveX = clamp1(mx);
    state.actions.moveY = clamp1(my);
    state.actions.lift = clamp1(lift);
    state.actions.sprint = keys.has("ShiftLeft") || keys.has("ShiftRight") || pad.sprint;
  };

  const requestFlyLock = () => {
    target.requestPointerLock?.();
  };

  const dispose = () => {
    target.removeEventListener("pointerdown", onPointerDown);
    target.removeEventListener("pointermove", onPointerMove);
    target.removeEventListener("pointerup", onPointerUp);
    target.removeEventListener("pointercancel", onPointerUp);
    target.removeEventListener("wheel", onWheel);
    window.removeEventListener("keydown", onKeyDown);
    window.removeEventListener("keyup", onKeyUp);
    window.removeEventListener("blur", clearKeys);
    document.removeEventListener("pointerlockchange", onLockChange);
  };

  return { state, sample, dispose, requestFlyLock };
}

function isTyping(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  const tag = t.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || t.isContentEditable;
}

function clamp1(v: number) {
  return Math.max(-1, Math.min(1, v));
}

function samplePad(): { x: number; y: number; sprint: boolean } {
  if (typeof navigator === "undefined" || !navigator.getGamepads) return { x: 0, y: 0, sprint: false };
  const pads = navigator.getGamepads();
  for (const p of pads) {
    if (!p || p.mapping !== "standard") continue;
    const x = p.axes[0] ?? 0;
    const y = -(p.axes[1] ?? 0);
    const m = Math.hypot(x, y);
    const dz = 0.15;
    let nx = 0, ny = 0;
    if (m >= dz) {
      const s = ((m - dz) / (1 - dz)) / m;
      nx = x * s;
      ny = y * s;
    }
    if (p.buttons[14]?.pressed) nx -= 1;
    if (p.buttons[15]?.pressed) nx += 1;
    if (p.buttons[12]?.pressed) ny += 1;
    if (p.buttons[13]?.pressed) ny -= 1;
    return { x: nx, y: ny, sprint: !!p.buttons[0]?.pressed };
  }
  return { x: 0, y: 0, sprint: false };
}
