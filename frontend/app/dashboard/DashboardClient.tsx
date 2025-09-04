// frontend/app/dashboard/DashboardClient.tsx
"use client";

import { useEffect, useMemo, useState } from "react";
import { Line, Scatter } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
} from "chart.js";
import { api } from "@/lib/api";
import type { CorrelationsResponse } from "@/types/api";
import type { ChartData, ChartOptions } from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend);

type Props = {
  initialCountry: string;
  countries: string[];
};

const controlBase: React.CSSProperties = {
  height: 36,
  padding: "6px 10px",
  borderRadius: 8,
  border: "1px solid #d0d0d0",
  backgroundColor: "#fff",
  color: "#111",
  fontSize: 14,
  outline: "none",
};

// ---------- stats helpers ----------
function pearson(xs: number[], ys: number[]): number | null {
  const n = Math.min(xs.length, ys.length);
  if (!n) return null;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, denx = 0, deny = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    num += dx * dy;
    denx += dx * dx;
    deny += dy * dy;
  }
  if (denx <= 0 || deny <= 0) return null;
  return num / Math.sqrt(denx * deny);
}

function linreg(xs: number[], ys: number[]) {
  const n = Math.min(xs.length, ys.length);
  if (!n) return { a: 0, b: 0 }; // y = a + b x
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  const b = sxx === 0 ? 0 : sxy / sxx;
  const a = my - b * mx;
  return { a, b };
}

export default function DashboardClient({ initialCountry, countries }: Props) {
  const [geo, setGeo] = useState<string>(initialCountry);
  const [windowMonths, setWindowMonths] = useState<number>(36);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [cpi, setCpi] = useState<[string, number | null][]>([]);
  const [spx, setSpx] = useState<[string, number | null][]>([]);
  const [corr, setCorr] = useState<CorrelationsResponse | null>(null);
  const [corrErr, setCorrErr] = useState<string | null>(null);

  async function loadSeries(geoArg: string) {
    setLoading(true);
    setErr(null);
    try {
      const [cpiRes, spxRes] = await Promise.all([
        api.series(geoArg, "cpi_yoy"),
        api.series(geoArg, "spx_ret"),
      ]);
      setCpi(cpiRes.series);
      setSpx(spxRes.series);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setLoading(false);
    }
  }

  async function loadCorr(window: number) {
    setCorrErr(null);
    try {
      const res = await api.correlations(window);
      setCorr(res);
    } catch (e: any) {
      setCorrErr(String(e?.message ?? e));
      setCorr(null);
    }
  }

  useEffect(() => { loadSeries(geo); }, [geo]);
  useEffect(() => { loadCorr(windowMonths); }, [windowMonths]);

  // Align dates and slice to window
  const { labels, cpiVals, spxVals } = useMemo(() => {
    const map = new Map<string, { c?: number | null; s?: number | null }>();
    for (const [d, v] of cpi) map.set(d, { ...(map.get(d) || {}), c: v });
    for (const [d, v] of spx) map.set(d, { ...(map.get(d) || {}), s: v });
    const allDates = Array.from(map.keys()).sort();
    const lastN = Math.max(1, windowMonths);
    const sliced = allDates.slice(-lastN);
    return {
      labels: sliced,
      cpiVals: sliced.map((k) => map.get(k)?.c ?? null),
      spxVals: sliced.map((k) => map.get(k)?.s ?? null),
    };
  }, [cpi, spx, windowMonths]);

  // ---------- Main time-series chart ----------
  const lineData = useMemo(
    () => ({
      labels,
      datasets: [
        { label: `CPI YoY (%) — ${geo}`, data: cpiVals, spanGaps: true },
        { label: "SPX Return (%)", data: spxVals, spanGaps: true },
      ],
    }),
    [labels, cpiVals, spxVals, geo]
  );

  const lineOptions = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false as const,
      interaction: { mode: "index" as const, intersect: false },
      scales: {
        x: { ticks: { maxRotation: 0, autoSkip: true, autoSkipPadding: 12, color: "#111" }, grid: { color: "rgba(0,0,0,0.06)" } },
        y: { beginAtZero: false, ticks: { color: "#111" }, grid: { color: "rgba(0,0,0,0.06)" } },
      },
      plugins: { legend: { position: "top" as const, labels: { color: "#111" } }, tooltip: { enabled: true } },
    }),
    []
  );

  // ---------- Correlation scatter (X=CPI, Y=SPX) ----------
  const { scatterData, scatterOptions, rLabel } = useMemo(() => {
    // Collect aligned numeric pairs
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < cpiVals.length; i++) {
      const x = cpiVals[i];
      const y = spxVals[i];
      if (x !== null && y !== null && Number.isFinite(x) && Number.isFinite(y)) {
        xs.push(x);
        ys.push(y);
      }
    }
    const r = pearson(xs, ys);
    const rTxt = r === null ? "n/a" : r.toFixed(2);

    // Regression line (as a scatter dataset with showLine to satisfy types)
    const { a, b } = linreg(xs, ys);
    const xmin = xs.length ? Math.min(...xs) : 0;
    const xmax = xs.length ? Math.max(...xs) : 1;
    const regLine = [
      { x: xmin, y: a + b * xmin },
      { x: xmax, y: a + b * xmax },
    ];

    const data: ChartData<"scatter", { x: number; y: number }[]> = {
      datasets: [
        {
          label: `CPI vs SPX — ${geo}`,
          data: xs.map((x, i) => ({ x, y: ys[i] })),
          pointRadius: 3,
        },
        {
          label: "OLS Fit",
          data: regLine,
          pointRadius: 0,
          showLine: true, // ← key change: keep dataset type "scatter" but draw a line
        },
      ],
    };

    const options: ChartOptions<"scatter"> = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#111" } }, tooltip: { enabled: true } },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "CPI YoY (%)", color: "#111" },
          ticks: { color: "#111" },
          grid: { color: "rgba(0,0,0,0.06)" },
        },
        y: {
          type: "linear",
          title: { display: true, text: "SPX Return (%)", color: "#111" },
          ticks: { color: "#111" },
          grid: { color: "rgba(0,0,0,0.06)" },
        },
      },
    };

    return { scatterData: data, scatterOptions: options, rLabel: `Pearson r: ${rTxt}` };
  }, [cpiVals, spxVals, geo]);

  // ===== Heatmap helpers (unchanged) =====
  function colorForCorr(v: number | null | undefined): string {
    if (v === null || v === undefined || Number.isNaN(v)) return "#eee";
    const clamp = Math.max(-1, Math.min(1, v));
    if (clamp >= 0) {
      const t = clamp;
      const r = Math.round(255 * 1);
      const g = Math.round(255 * (1 - 0.6 * t));
      const b = Math.round(255 * (1 - 0.6 * t));
      return `rgb(${r},${g},${b})`;
    } else {
      const t = -clamp;
      const r = Math.round(255 * (1 - 0.6 * t));
      const g = Math.round(255 * (1 - 0.6 * t));
      const b = Math.round(255 * 1);
      return `rgb(${r},${g},${b})`;
    }
  }

  const Heatmap = useMemo(() => {
    if (!corr) return null;
    const maxCols = 12;
    const totalCols = corr.dates.length;
    const startCol = Math.max(0, totalCols - maxCols);
    const shownDates = corr.dates.slice(startCol);

    return (
      <div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 8 }}>
          <strong style={{ color: "#111" }}>CPI ↔ SPX rolling correlation</strong>
          <span style={{ color: "#666" }}>(window: {corr.window_months}m, last {maxCols} months)</span>
        </div>

        {/* Dates header */}
        <div style={{ marginLeft: 160, display: "flex", gap: 2, marginBottom: 6 }}>
          {shownDates.map((d) => (
            <div key={d} style={{ width: 22, fontSize: 10, color: "#666", textAlign: "center" }}>
              {d.slice(2, 7)}
            </div>
          ))}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "160px auto", gap: 8 }}>
          {/* Country labels */}
          <div style={{ display: "grid", gap: 4 }}>
            {corr.countries.map((country) => (
              <div key={country} style={{ color: "#111", fontSize: 12, lineHeight: "22px", height: 22, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {country}
              </div>
            ))}
          </div>

        {/* Heat grid */}
          <div style={{ display: "grid", gap: 4 }}>
            {corr.countries.map((country, rIdx) => (
              <div key={country} style={{ display: "flex" }}>
                {corr.dates.slice(startCol).map((_, cOff) => {
                  const cIdx = startCol + cOff;
                  const v = corr.matrix[cIdx]?.[rIdx] ?? null;
                  return (
                    <div
                      key={`${rIdx}-${cIdx}`}
                      title={`${country} @ ${corr.dates[cIdx]} → ${v === null || Number.isNaN(v) ? "NA" : v.toFixed(2)}`}
                      style={{
                        width: 22,
                        height: 22,
                        background: colorForCorr(v),
                        border: "1px solid #fff",
                        boxSizing: "border-box",
                      }}
                    />
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        {/* Legend */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8 }}>
          <span style={{ fontSize: 12, color: "#666" }}>−1</span>
          <div style={{ display: "flex" }}>
            {[-1, -0.5, 0, 0.5, 1].map((v) => (
              <div key={v} style={{ width: 32, height: 10, background: colorForCorr(v) }} />
            ))}
          </div>
          <span style={{ fontSize: 12, color: "#666" }}>+1</span>
        </div>
      </div>
    );
  }, [corr]);

  return (
    <section style={{ display: "grid", gap: 16 }}>
      {/* Controls */}
      <div
        style={{
          display: "flex",
          gap: 12,
          flexWrap: "wrap",
          alignItems: "center",
          border: "1px solid #e6e6e6",
          padding: 12,
          borderRadius: 12,
          background: "#fafafa",
          color: "#111",
        }}
      >
        <label style={{ color: "#111", fontWeight: 600 }}>
          Country:&nbsp;
          <select
            value={geo}
            onChange={(e) => setGeo(e.target.value)}
            style={{ ...controlBase, minWidth: 220 }}
          >
            {countries.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>

        <label style={{ color: "#111", fontWeight: 600 }}>
          Corr Window (months):&nbsp;
          <select
            value={windowMonths}
            onChange={(e) => setWindowMonths(parseInt(e.target.value, 10))}
            style={{ ...controlBase, minWidth: 140 }}
          >
            {[12, 24, 36, 60, 120].map((w) => (
              <option key={w} value={w}>
                {w}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* 1) Time-series chart */}
      <div
        key={`${geo}-${windowMonths}-line`}
        style={{
          height: 420,
          border: "1px solid #e6e6e6",
          borderRadius: 12,
          padding: 12,
          background: "#ffffff",
        }}
      >
        {loading ? (
          <div style={{ color: "#111" }}>Loading chart…</div>
        ) : err ? (
          <div style={{ color: "crimson" }}>Error: {err}</div>
        ) : (
          <Line data={lineData} options={lineOptions} />
        )}
      </div>

      {/* 2) Correlation scatter */}
      <div
        key={`${geo}-${windowMonths}-scatter`}
        style={{
          height: 420,
          border: "1px solid #e6e6e6",
          borderRadius: 12,
          padding: 12,
          background: "#ffffff",
          color: "#111",
        }}
      >
        <div style={{ marginBottom: 8, fontWeight: 700 }}>{rLabel}</div>
        {loading ? (
          <div style={{ color: "#111" }}>Loading scatter…</div>
        ) : err ? (
          <div style={{ color: "crimson" }}>Error: {err}</div>
        ) : (
          <Scatter data={scatterData} options={scatterOptions} />
        )}
      </div>

      {/* 3) Heatmap */}
      <div
        style={{
          border: "1px solid #e6e6e6",
          borderRadius: 12,
          padding: 12,
          background: "#ffffff",
          color: "#111",
          maxHeight: 520,
          overflow: "auto",
        }}
      >
        {corrErr ? (
          <div style={{ color: "crimson" }}>Error: {corrErr}</div>
        ) : !corr ? (
          <div>Loading correlations…</div>
        ) : (
          Heatmap
        )}
      </div>
    </section>
  );
}
