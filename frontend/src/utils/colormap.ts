function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

function rgba(r: number, g: number, b: number, a: number): string {
  return `rgba(${r},${g},${b},${a})`;
}

export function concentrationColor(c: number): string {
  const v = Math.max(0, Math.min(1, c));
  if (v < 0.15) return rgba(255, 255, 255, 0.05);
  if (v < 0.3) return rgba(lerp(200, 150, (v - 0.15) / 0.15), lerp(230, 200, (v - 0.15) / 0.15), 255, 0.55);
  if (v < 0.5) return rgba(100, 180, 240, 0.65);
  if (v < 0.7) return rgba(30, 130, 220, 0.75);
  if (v < 0.85) return rgba(10, 80, 180, 0.82);
  return rgba(0, 40, 120, 0.88);
}

export function concentrationToCanvas(
  conc: number[][],
): HTMLCanvasElement {
  const rows = conc.length;
  const cols = conc[0]?.length ?? 0;
  const canvas = document.createElement("canvas");
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext("2d")!;
  for (let r = 0; r < rows; r++) {
    const canvasRow = rows - 1 - r;
    for (let c = 0; c < cols; c++) {
      ctx.fillStyle = concentrationColor(conc[r][c]);
      ctx.fillRect(c, canvasRow, 1, 1);
    }
  }
  return canvas;
}

export function seaIceLegendStops(): { pct: number; color: string }[] {
  return [
    { pct: 0, color: "rgba(255,255,255,0.05)" },
    { pct: 15, color: "rgba(200,230,255,0.55)" },
    { pct: 30, color: "rgba(100,180,240,0.65)" },
    { pct: 50, color: "rgba(30,130,220,0.75)" },
    { pct: 70, color: "rgba(10,80,180,0.82)" },
    { pct: 100, color: "rgba(0,40,120,0.88)" },
  ];
}