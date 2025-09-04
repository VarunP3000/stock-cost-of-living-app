"use client";

import { useEffect, useMemo, useState } from "react";
import { Line } from "react-chartjs-2";
import Link from "next/link";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

// ---- Types matching backend realtime endpoints ----
type RealtimeModels = { models: string[]; artifact_dir: string };
type PastPanelReq = { start?: string | null; end?: string | null; weights?: Record<string, number> | null };
type PastPanelResp = {
  points: { asof: string; ensemble: number | null; actual: number | null; model_std: number | null }[];
  metrics: { rmse: number | null; mae: number | null; mape: number | null; hit_rate: number | null };
  model_weights: Record<string, number>;
  model_list: string[];
  model_panel: ({ asof: string } & Record<string, number | null>)[];
};
type FuturePanelReq = { horizon: number; weights?: Record<string, number> | null };
type FuturePanelResp = {
  forecast: { asof: string; yhat: number | null; disagreement: number | null }[];
  confidence_band: { asof: string; yhat_lo: number | null; yhat_hi: number | null }[];
  model_weights: Record<string, number>;
  model_list: string[];
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://127.0.0.1:8000";

export default function CustomWeightsPage() {
  const [models, setModels] = useState<string[]>([]);
  const [weights, setWeights] = useState<Record<string, number>>({});
  const [start, setStart] = useState<string>("");
  const [end, setEnd] = useState<string>("");
  const [horizon, setHorizon] = useState<number>(6);

  const [past, setPast] = useState<PastPanelResp | null>(null);
  const [future, setFuture] = useState<FuturePanelResp | null>(null);
  const [loadingPast, setLoadingPast] = useState(false);
  const [loadingFuture, setLoadingFuture] = useState(false);
  const [errPast, setErrPast] = useState<string | null>(null);
  const [errFuture, setErrFuture] = useState<string | null>(null);

  // Fetch available predictor models
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/realtime/models`);
        if (!r.ok) throw new Error(`GET /realtime/models -> ${r.status}`);
        const j: RealtimeModels = await r.json();
        setModels(j.models);
        // default equal weights
        const w: Record<string, number> = {};
        j.models.forEach((m) => (w[m] = 1 / Math.max(1, j.models.length)));
        setWeights(w);
      } catch (e: any) {
        console.error(e);
      }
    })();
  }, []);

  // Normalize weights to sum to 1
  const normalizedWeights = useMemo(() => {
    const s = Object.values(weights).reduce((a, b) => a + (isFinite(b) ? b : 0), 0);
    if (s <= 0) return weights;
    const out: Record<string, number> = {};
    for (const k of Object.keys(weights)) out[k] = weights[k] / s;
    return out;
  }, [weights]);

  function setWeight(name: string, val: number) {
    setWeights((prev) => ({ ...prev, [name]: val }));
  }

  // Past panel fetch
  async function runPast() {
    try {
      setErrPast(null);
      setLoadingPast(true);
      const body: PastPanelReq = {
        start: start || undefined,
        end: end || undefined,
        weights: normalizedWeights,
      };
      const r = await fetch(`${API_BASE}/realtime/past_panel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const text = await r.text();
        throw new Error(`POST /realtime/past_panel -> ${r.status} ${text}`);
      }
      setPast(await r.json());
    } catch (e: any) {
      setErrPast(e?.message ?? String(e));
      setPast(null);
    } finally {
      setLoadingPast(false);
    }
  }

  // Future panel fetch
  async function runFuture() {
    try {
      setErrFuture(null);
      setLoadingFuture(true);
      const body: FuturePanelReq = {
        horizon,
        weights: normalizedWeights,
      };
      const r = await fetch(`${API_BASE}/realtime/future_panel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const text = await r.text();
        throw new Error(`POST /realtime/future_panel -> ${r.status} ${text}`);
      }
      setFuture(await r.json());
    } catch (e: any) {
      setErrFuture(e?.message ?? String(e));
      setFuture(null);
    } finally {
      setLoadingFuture(false);
    }
  }

  // --- History chart (ensemble vs actual + implicit band via model_std as shading) ---
  const pastChart = useMemo(() => {
    if (!past) return null;
    const labels = past.points.map((p) => p.asof);
    const ens = past.points.map((p) => (p.ensemble ?? null));
    const act = past.points.map((p) => (p.actual ?? null));
    const std = past.points.map((p) => (p.model_std ?? 0));

    // Confidence-like band from ensemble ± model_std (for visualization)
    const up = ens.map((y, i) => (y != null && std[i] != null ? (y as number) + (std[i] as number) : null));
    const dn = ens.map((y, i) => (y != null && std[i] != null ? (y as number) - (std[i] as number) : null));

    const datasets: any[] = [
      { label: "Band Upper", data: up, borderWidth: 0, pointRadius: 0, fill: "-1" as const, backgroundColor: "rgba(100,149,237,0.18)" },
      { label: "Band Lower", data: dn, borderWidth: 0, pointRadius: 0, fill: false },
      { label: "Ensemble (past)", data: ens, pointRadius: 0 },
    ];
    if (act.some((v) => v !== null)) {
      datasets.push({ label: "Actual", data: act, pointRadius: 0 });
    }

    return {
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index" as const, intersect: false },
        scales: {
          x: { ticks: { color: "#111" }, grid: { color: "rgba(0,0,0,0.06)" } },
          y: { ticks: { color: "#111" }, grid: { color: "rgba(0,0,0,0.06)" } },
        },
        plugins: { legend: { labels: { color: "#111" } }, tooltip: { enabled: true } },
      },
    };
  }, [past]);

  // --- Future chart (yhat + band) ---
  const futureChart = useMemo(() => {
    if (!future) return null;
    const labels = future.forecast.map((f) => f.asof);
    const yhat = future.forecast.map((f) => f.yhat ?? null);
    const lo = future.confidence_band.map((b) => b.yhat_lo ?? null);
    const hi = future.confidence_band.map((b) => b.yhat_hi ?? null);

    return {
      data: {
        labels,
        datasets: [
          { label: "Band Upper", data: hi, borderWidth: 0, pointRadius: 0, fill: "-1" as const, backgroundColor: "rgba(100,149,237,0.18)" },
          { label: "Band Lower", data: lo, borderWidth: 0, pointRadius: 0, fill: false },
          { label: "Future Prediction", data: yhat, spanGaps: true, pointRadius: 2 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index" as const, intersect: false },
        scales: {
          x: { ticks: { color: "#111" }, grid: { color: "rgba(0,0,0,0.06)" } },
          y: { ticks: { color: "#111" }, grid: { color: "rgba(0,0,0,0.06)" } },
        },
        plugins: { legend: { labels: { color: "#111" } }, tooltip: { enabled: true } },
      },
    };
  }, [future]);

  // Pretty print metrics
  const m = past?.metrics;
  const metricsLine = m
    ? `RMSE: ${fmt(m.rmse)} · MAE: ${fmt(m.mae)} · MAPE: ${fmtPct(m.mape)} · Dir Hit: ${fmtPct(m.hit_rate)}`
    : null;

  return (
    <main style={{ minHeight: "100vh", background: "#0b0b0b", padding: 24 }}>
      <div style={card()}>
        <div style={{ marginBottom: 12 }}>
          <Link href="/" style={btn()}>← Back</Link>
        </div>

        <h1 style={h1()}>Forecast — Custom Ensemble Weights</h1>
        <p style={{ margin: "0 0 8px", color: "#333" }}>
          Past: ensemble vs actual (+ band). Future: forecast (+ band). Adjust model weights and recompute.
        </p>

        {/* Weight sliders */}
        <div style={{ ...ctrlBar(), display: "grid", gap: 8 }}>
          <div style={{ fontWeight: 700, color: "#111" }}>Weights (sum=1 after normalization):</div>
          {models.length === 0 ? (
            <div style={{ color: "#666" }}>No predictor models found.</div>
          ) : (
            models.map((name) => (
              <div key={name} style={{ display: "grid", gridTemplateColumns: "240px 1fr 90px", gap: 10, alignItems: "center" }}>
                <div style={{ color: "#111", fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis" }}>{name}</div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={weights[name] ?? 0}
                  onChange={(e) => setWeight(name, parseFloat(e.target.value))}
                />
                <div style={{ textAlign: "right", color: "#111" }}>{((normalizedWeights[name] ?? 0) * 100).toFixed(0)}%</div>
              </div>
            ))
          )}
        </div>

        {/* History controls */}
        <div style={{ ...ctrlBar(), marginTop: 12 }}>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <label style={ctrlLabel()}>
              Start (YYYY-MM-DD):{" "}
              <input type="text" placeholder="e.g. 2015-01-01" value={start} onChange={(e) => setStart(e.target.value)} style={input()} />
            </label>
            <label style={ctrlLabel()}>
              End (YYYY-MM-DD):{" "}
              <input type="text" placeholder="e.g. 2024-12-01" value={end} onChange={(e) => setEnd(e.target.value)} style={input()} />
            </label>
            <button onClick={runPast} style={btn()} disabled={loadingPast}>
              {loadingPast ? "Computing…" : "Update Past + Metrics"}
            </button>
          </div>
        </div>

        {/* Past chart */}
        <h3 style={{ ...h3(), marginTop: 10 }}>Past: Predictions vs Actuals</h3>
        {errPast ? (
          <div style={{ color: "crimson", marginTop: 6 }}>Error: {errPast}</div>
        ) : past ? (
          <>
            <div style={{ marginTop: 6, color: "#333" }}>{metricsLine}</div>
            <div style={chartBox()}>
              <Line data={pastChart!.data} options={pastChart!.options as any} />
            </div>
          </>
        ) : (
          <div style={{ color: "#666" }}>Run “Update Past + Metrics” to populate.</div>
        )}

        {/* Future controls */}
        <div style={{ ...ctrlBar(), marginTop: 16 }}>
          <label style={ctrlLabel()}>
            Future horizon (months):{" "}
            <select value={horizon} onChange={(e) => setHorizon(parseInt(e.target.value, 10))} style={selectStyle()}>
              {[3, 6, 9, 12, 18, 24].map((h) => <option key={h} value={h}>{h}</option>)}
            </select>
          </label>
          <button onClick={runFuture} style={{ ...btn(), marginLeft: 12 }} disabled={loadingFuture}>
            {loadingFuture ? "Computing…" : "Generate Future"}
          </button>
        </div>

        {/* Future chart */}
        <h3 style={{ ...h3(), marginTop: 10 }}>Future: Forecast</h3>
        {errFuture ? (
          <div style={{ color: "crimson" }}>Error: {errFuture}</div>
        ) : future ? (
          <div style={chartBox()}>
            <Line data={futureChart!.data} options={futureChart!.options as any} />
          </div>
        ) : (
          <div style={{ color: "#666" }}>Run “Generate Future” to populate.</div>
        )}
      </div>
    </main>
  );
}

// ---- helpers ----
function fmt(x?: number | null) {
  return x == null || !isFinite(x) ? "—" : Number(x).toFixed(4);
}
function fmtPct(x?: number | null) {
  return x == null || !isFinite(x) ? "—" : (Number(x) * 100).toFixed(1) + "%";
}

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
