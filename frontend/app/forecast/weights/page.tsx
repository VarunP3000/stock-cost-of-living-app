// frontend/app/forecast/weights/page.tsx
"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
  type ChartData,
  type ChartOptions,
  type ChartDataset,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

// ---------- Response shapes from your backend ----------
type Past = {
  asof: string;
  dates: string[];
  prediction: (number | null)[];
  actual: (number | null)[];
  p10: (number | null)[];
  p90: (number | null)[];
  models: string[];
  weights: number[];
};

type Future = {
  asof: string;
  future_dates: string[];
  future_prediction: (number | null)[];
  p10: (number | null)[];
  p90: (number | null)[];
  models: string[];
  weights: number[];
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://127.0.0.1:8000";

// ---------- helpers ----------
function normalizeWeights(raw: Record<string, number>): Record<string, number> {
  const clamped: Record<string, number> = {};
  for (const [k, v] of Object.entries(raw)) clamped[k] = Math.max(0, Number(v) || 0);
  const s = Object.values(clamped).reduce((a, b) => a + b, 0);
  if (s <= 0) {
    const keys = Object.keys(raw);
    const eq = 1 / Math.max(1, keys.length);
    return Object.fromEntries(keys.map((k) => [k, eq]));
  }
  return Object.fromEntries(Object.entries(clamped).map(([k, v]) => [k, v / s]));
}

const _toPct = (x: number) => Math.round(x * 100); // kept, prefixed to avoid "unused" rule

const hasAny = (arr?: (number | null)[]): arr is number[] =>
  Array.isArray(arr) && arr.some((v): v is number => typeof v === "number" && !Number.isNaN(v));

// ---------- component ----------
export default function CustomWeightsPage() {
  const [past, setPast] = useState<Past | null>(null);
  const [future, setFuture] = useState<Future | null>(null);
  const [errPast, setErrPast] = useState<string | null>(null);
  const [errFuture, setErrFuture] = useState<string | null>(null);

  // controls
  const [start, setStart] = useState<string>("2015-01-01");
  const [end, setEnd] = useState<string>("2025-01-01");
  const [hzn, setHzn] = useState<number>(6);
  const [weights, setWeights] = useState<Record<string, number>>({}); // fractions (0..1)

  // init: fetch default models/weights
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/ensemble/past`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ start, end }),
        });
        if (!r.ok) throw new Error(`POST /ensemble/past -> ${r.status} ${await r.text()}`);
        const data: Past = await r.json();
        setPast(data);

        if (data.models?.length) {
          const equal = Object.fromEntries(data.models.map((m) => [m, 1 / data.models.length]));
          setWeights(equal);
        }
      } catch (e: unknown) {
        setErrPast(e instanceof Error ? e.message : String(e));
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const updatePast = useCallback(async () => {
    try {
      setErrPast(null);
      const body = { start, end, weights: normalizeWeights(weights) };
      const r = await fetch(`${API_BASE}/ensemble/past`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`POST /ensemble/past -> ${r.status} ${await r.text()}`);
      const data: Past = await r.json();
      setPast(data);
    } catch (e: unknown) {
      setErrPast(e instanceof Error ? e.message : String(e));
      setPast(null);
    }
  }, [start, end, weights]);

  const genFuture = useCallback(async () => {
    try {
      setErrFuture(null);
      const r = await fetch(`${API_BASE}/ensemble/future`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ horizon: hzn, weights: normalizeWeights(weights) }),
      });
      if (!r.ok) throw new Error(`POST /ensemble/future -> ${r.status} ${await r.text()}`);
      const data: Future = await r.json();
      setFuture(data);
    } catch (e: unknown) {
      setErrFuture(e instanceof Error ? e.message : String(e));
      setFuture(null);
    }
  }, [hzn, weights]);

  // ---------- metrics over the visible window ----------
  const metrics = useMemo(() => {
    if (!past) return null;
    const yhat = past.prediction;
    const y = past.actual;
    if (!hasAny(yhat) || !hasAny(y)) return null;

    let n = 0,
      sq = 0,
      abs = 0,
      apeSum = 0,
      dirHit = 0;
    let prevY: number | null = null;
    let prevYhat: number | null = null;

    for (let i = 0; i < yhat.length; i++) {
      const yh = yhat[i];
      const ya = y[i];
      if (typeof yh !== "number" || typeof ya !== "number" || Number.isNaN(yh) || Number.isNaN(ya)) continue;
      const e = yh - ya;
      sq += e * e;
      abs += Math.abs(e);
      if (Math.abs(ya) > 1e-9) apeSum += Math.abs(e / ya);
      if (prevY != null && prevYhat != null) {
        const dAct = ya - prevY;
        const dPred = yh - prevYhat;
        if (Math.sign(dAct) === Math.sign(dPred)) dirHit += 1;
      }
      prevY = ya;
      prevYhat = yh;
      n += 1;
    }
    if (n === 0) return null;
    return {
      rmse: Math.sqrt(sq / n),
      mae: abs / n,
      mape: (apeSum / n) * 100,
      dirHit: (dirHit / Math.max(1, n - 1)) * 100,
    };
  }, [past]);

  // ---------- charts (typed) ----------
  type YArray = (number | null)[];
  type LineDS = ChartDataset<"line", YArray>;

  const pastChart = useMemo<{
    data: ChartData<"line", YArray, string>;
    options: ChartOptions<"line">;
  } | null>(() => {
    if (!past) return null;

    const labels = past.dates;
    const datasets: LineDS[] = [
      // In Chart.js v4, `fill` can be boolean | number | 'start'|'end'|'origin'
      { label: "p90", data: past.p90, borderWidth: 0, pointRadius: 0, fill: "origin", backgroundColor: "rgba(100,149,237,0.18)", spanGaps: true },
      { label: "p10", data: past.p10, borderWidth: 0, pointRadius: 0, fill: false, spanGaps: true },
      { label: "Ensemble (past)", data: past.prediction, pointRadius: 0, spanGaps: true },
    ];
    if (hasAny(past.actual)) {
      datasets.push({ label: "Actual", data: past.actual, pointRadius: 0, spanGaps: true });
    }

    const options: ChartOptions<"line"> = {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { ticks: { color: "#111" }, grid: { color: "rgba(0,0,0,0.06)" } },
        y: { ticks: { color: "#111" }, grid: { color: "rgba(0,0,0,0.06)" } },
      },
      plugins: { legend: { labels: { color: "#111" } }, tooltip: { enabled: true } },
    };

    return { data: { labels, datasets }, options };
  }, [past]);

  const futureChart = useMemo<{
    data: ChartData<"line", YArray, string>;
    options: ChartOptions<"line">;
  } | null>(() => {
    if (!future) return null;

    const datasets: LineDS[] = [
      { label: "Band Upper", data: future.p90, borderWidth: 0, pointRadius: 0, fill: "origin", backgroundColor: "rgba(100,149,237,0.18)", spanGaps: true },
      { label: "Band Lower", data: future.p10, borderWidth: 0, pointRadius: 0, fill: false, spanGaps: true },
      { label: "Future Prediction", data: future.future_prediction, pointRadius: 2, spanGaps: true },
    ];

    const options: ChartOptions<"line"> = {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#111" }, grid: { color: "rgba(0,0,0,0.06)" } },
        y: { ticks: { color: "#111" }, grid: { color: "rgba(0,0,0,0.06)" } },
      },
      plugins: { legend: { labels: { color: "#111" } }, tooltip: { enabled: true } },
    };

    return {
      data: { labels: future.future_dates, datasets },
      options,
    };
  }, [future]);

  // ---------- render ----------
  const modelList = past?.models ?? Object.keys(weights);
  const norm = normalizeWeights(weights);

  return (
    <main style={{ minHeight: "100vh", background: "#0b0b0b", padding: 24 }}>
      <div style={card()}>
        <div style={{ marginBottom: 12, display: "flex", gap: 8 }}>
          <Link href="/" style={btn()}>← Back</Link>
          <Link href="/forecast" style={btn()}>Standard forecast page →</Link>
        </div>

        <h1 style={h1()}>Forecast — Custom Ensemble Weights</h1>
        <p style={{ margin: "0 0 8px", color: "#333" }}>
          Past: <strong>ensemble vs actual</strong> (+band). Future: <strong>forecast</strong> (+band).
          Adjust model weights and recompute.
        </p>

        {/* weights editor */}
        <div style={{ ...ctrlBar(), flexWrap: "wrap" }}>
          <div style={{ fontWeight: 700, color: "#111" }}>Weights:</div>
          {modelList.map((m) => (
            <label key={m} style={{ display: "flex", alignItems: "center", gap: 6, color: "#111" }}>
              <span style={{ minWidth: 120 }}>{m}</span>
              <input
                type="number"
                min={0}
                max={100}
                step="1"
                value={Math.round((weights[m] ?? norm[m] ?? 0) * 100)}
                onChange={(e) =>
                  setWeights((w) => ({ ...w, [m]: Math.max(0, Math.min(100, Number(e.target.value) || 0)) / 100 }))
                }
                style={numInput()}
              />
              <span>%</span>
            </label>
          ))}
        </div>

        {/* past controls */}
        <div style={ctrlBar()}>
          <label style={ctrlLabel()}>
            Start (YYYY-MM-DD):&nbsp;
            <input type="text" placeholder="e.g. 2015-01-01" value={start} onChange={(e) => setStart(e.target.value)} style={input()} />
          </label>
          <label style={ctrlLabel()}>
            End (YYYY-MM-DD):&nbsp;
            <input type="text" placeholder="e.g. 2025-01-01" value={end} onChange={(e) => setEnd(e.target.value)} style={input()} />
          </label>
          <button onClick={updatePast} style={btn()}>Update Past + Metrics</button>
        </div>

        {/* metrics */}
        {metrics && (
          <div style={{ margin: "8px 0 0", color: "#333" }}>
            <strong>RMSE:</strong> {metrics.rmse.toFixed(4)} &nbsp;·&nbsp;
            <strong>MAE:</strong> {metrics.mae.toFixed(4)} &nbsp;·&nbsp;
            <strong>MAPE:</strong> {metrics.mape.toFixed(1)}% &nbsp;·&nbsp;
            <strong>Dir Hit:</strong> {metrics.dirHit.toFixed(1)}%
          </div>
        )}

        {/* past chart */}
        <h3 style={{ ...h3(), marginTop: 10 }}>Past: Predictions vs Actuals</h3>
        {errPast ? (
          <div style={{ color: "crimson" }}>Error: {errPast}</div>
        ) : !past || !pastChart ? (
          <div>Loading…</div>
        ) : (
          <div style={chartBox()}>
            <Line data={pastChart.data} options={pastChart.options} />
          </div>
        )}

        {/* future controls */}
        <div style={{ ...ctrlBar(), marginTop: 18 }}>
          <label style={ctrlLabel()}>
            Future horizon (months):&nbsp;
            <select value={hzn} onChange={(e) => setHzn(parseInt(e.target.value, 10))} style={selectStyle()}>
              {[3, 6, 9, 12, 18, 24].map((h) => <option key={h} value={h}>{h}</option>)}
            </select>
          </label>
          <button onClick={genFuture} style={btn()}>Generate Future</button>
        </div>

        {/* future chart */}
        <h3 style={{ ...h3(), marginTop: 10 }}>Future: Forecast</h3>
        {errFuture ? (
          <div style={{ color: "crimson" }}>Error: {errFuture}</div>
        ) : !future || !futureChart ? (
          <div>Loading…</div>
        ) : (
          <div style={chartBox()}>
            <Line data={futureChart.data} options={futureChart.options} />
          </div>
        )}

        {/* meta */}
        <div style={{ marginTop: 12, color: "#333" }}>
          {past?.asof && <span><strong>Past as of:</strong> {new Date(past.asof).toLocaleString()}</span>}
          {future?.asof && <span style={{ marginLeft: 12 }}><strong>Future as of:</strong> {new Date(future.asof).toLocaleString()}</span>}
        </div>
      </div>
    </main>
  );
}

// ---------- styles ----------
const card = () => ({
  maxWidth: 1100,
  margin: "0 auto",
  background: "#fff",
  color: "#111",
  borderRadius: 16,
  border: "1px solid #e6e6e6",
  padding: 20,
  boxShadow: "0 8px 24px rgba(0,0,0,0.1)",
});
const h1 = () => ({ fontSize: 28, margin: "4px 0 8px", fontWeight: 800, color: "#111" });
const h3 = () => ({ fontSize: 18, margin: "0 0 4px", fontWeight: 800, color: "#111" });
const btn = () => ({
  display: "inline-block",
  padding: "8px 12px",
  borderRadius: 10,
  border: "1px solid #d0d0d0",
  textDecoration: "none",
  background: "#f8f8f8",
  color: "#111",
  fontWeight: 600,
  cursor: "pointer",
});
const ctrlBar = () => ({
  display: "flex",
  gap: 12,
  alignItems: "center",
  border: "1px solid #e6e6e6",
  padding: 12,
  borderRadius: 12,
  background: "#fafafa",
  color: "#111",
});
const ctrlLabel = () => ({ color: "#111", fontWeight: 600 });
const input = () => ({
  height: 36,
  padding: "6px 10px",
  borderRadius: 8,
  border: "1px solid #d0d0d0",
  backgroundColor: "#fff",
  color: "#111",
  fontSize: 14,
});
const numInput = () => ({
  width: 80,
  height: 36,
  padding: "6px 10px",
  borderRadius: 8,
  border: "1px solid #d0d0d0",
  backgroundColor: "#fff",
  color: "#111",
  fontSize: 14,
});
const selectStyle = () => ({
  height: 36,
  padding: "6px 10px",
  borderRadius: 8,
  border: "1px solid #d0d0d0",
  backgroundColor: "#fff",
  color: "#111",
  fontSize: 14,
});
const chartBox = () => ({
  height: 420,
  border: "1px solid #e6e6e6",
  borderRadius: 12,
  padding: 12,
  marginTop: 8,
});
