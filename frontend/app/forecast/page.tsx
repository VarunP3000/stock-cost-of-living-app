// frontend/app/forecast/page.tsx
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState, useCallback } from "react";
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

export default function ForecastPage() {
  const [past, setPast] = useState<Past | null>(null);
  const [future, setFuture] = useState<Future | null>(null);
  const [errPast, setErrPast] = useState<string | null>(null);
  const [errFuture, setErrFuture] = useState<string | null>(null);

  // controls
  const [start, setStart] = useState<string>("");
  const [end, setEnd] = useState<string>("");
  const [hzn, setHzn] = useState<number>(6);

  const loadPast = useCallback(async () => {
    try {
      setErrPast(null);
      const body: Record<string, string> = {};
      if (start) body.start = start;
      if (end) body.end = end;
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
  }, [start, end]);

  const loadFuture = useCallback(async () => {
    try {
      setErrFuture(null);
      const r = await fetch(`${API_BASE}/ensemble/future`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ horizon: hzn }),
      });
      if (!r.ok) throw new Error(`POST /ensemble/future -> ${r.status} ${await r.text()}`);
      const data: Future = await r.json();
      setFuture(data);
    } catch (e: unknown) {
      setErrFuture(e instanceof Error ? e.message : String(e));
      setFuture(null);
    }
  }, [hzn]);

  useEffect(() => {
    void loadPast();
  }, [loadPast]);

  useEffect(() => {
    void loadFuture();
  }, [loadFuture]);

  const hasAny = (arr?: (number | null)[]): arr is number[] =>
    Array.isArray(arr) && arr.some((v): v is number => typeof v === "number" && !Number.isNaN(v));

  // --- PAST chart (prediction vs actual + band) ---
  type YArray = (number | null)[];
  type LineDS = ChartDataset<"line", YArray>;

  const pastChart = useMemo<{
    data: ChartData<"line", YArray, string>;
    options: ChartOptions<"line">;
  } | null>(() => {
    if (!past) return null;

    const labels = past.dates;
    const datasets: LineDS[] = [
      { label: "p90", data: past.p90, borderWidth: 0, pointRadius: 0, fill: "origin", backgroundColor: "rgba(100,149,237,0.18)", spanGaps: true },
      { label: "p10", data: past.p10, borderWidth: 0, pointRadius: 0, fill: false, spanGaps: true },
      { label: "Prediction", data: past.prediction, spanGaps: true, pointRadius: 0 },
    ];
    if (hasAny(past.actual)) {
      datasets.push({ label: "Actual", data: past.actual, spanGaps: true, pointRadius: 0 });
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

  // --- FUTURE chart (yhat + band) ---
  const futureChart = useMemo<{
    data: ChartData<"line", YArray, string>;
    options: ChartOptions<"line">;
  } | null>(() => {
    if (!future) return null;
    const datasets: LineDS[] = [
      { label: "p90", data: future.p90, borderWidth: 0, pointRadius: 0, fill: "origin", backgroundColor: "rgba(100,149,237,0.18)", spanGaps: true },
      { label: "p10", data: future.p10, borderWidth: 0, pointRadius: 0, fill: false, spanGaps: true },
      { label: "Future Prediction", data: future.future_prediction, spanGaps: true, pointRadius: 2 },
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

    return { data: { labels: future.future_dates, datasets }, options };
  }, [future]);

  const ensembleLabel = useMemo(() => {
    if (!past?.models || !past?.weights || past.models.length !== past.weights.length) return null;
    return past.models.map((m, i) => `${m}=${(past.weights[i] * 100).toFixed(0)}%`).join(" · ");
  }, [past]);

  const futureLabel = useMemo(() => {
    if (!future?.models || !future?.weights || future.models.length !== future.weights.length) return null;
    return future.models.map((m, i) => `${m}=${(future.weights[i] * 100).toFixed(0)}%`).join(" · ");
  }, [future]);

  const futureHasLine = hasAny(future?.future_prediction);

  return (
    <main style={{ minHeight: "100vh", background: "#0b0b0b", padding: 24 }}>
      <div style={card()}>
        <div style={{ marginBottom: 12, display: "flex", gap: 8 }}>
          <Link href="/" style={btn()}>
            ← Back
          </Link>
          <Link href="/forecast/weights" style={btn()}>
            Try custom weights →
          </Link>
        </div>

        <h1 style={h1()}>Forecast</h1>
        <p style={{ margin: "0 0 8px", color: "#333" }}>
          Past: <strong>ensemble vs actual</strong> (+band). Future: <strong>forecast</strong> (+band).
        </p>

        {ensembleLabel && (
          <div style={{ margin: "0 0 8px", color: "#555", fontSize: 13 }}>Past Ensemble: {ensembleLabel}</div>
        )}
        {futureLabel && (
          <div style={{ margin: "0 0 8px", color: "#555", fontSize: 13 }}>Future Ensemble: {futureLabel}</div>
        )}

        {/* PAST controls */}
        <div style={ctrlBar()}>
          <label style={ctrlLabel()}>
            Start (YYYY-MM-DD):&nbsp;
            <input
              type="text"
              placeholder="e.g. 2015-01-01"
              value={start}
              onChange={(e) => setStart(e.target.value)}
              style={input()}
            />
          </label>
          <label style={ctrlLabel()}>
            End (YYYY-MM-DD):&nbsp;
            <input
              type="text"
              placeholder="e.g. 2024-12-01"
              value={end}
              onChange={(e) => setEnd(e.target.value)}
              style={input()}
            />
          </label>
          <button onClick={loadPast} style={btn()}>
            Update Past
          </button>
        </div>

        {/* PAST chart */}
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

        {/* FUTURE controls */}
        <div style={{ ...ctrlBar(), marginTop: 18 }}>
          <label style={ctrlLabel()}>
            Future horizon (months):&nbsp;
            <select value={hzn} onChange={(e) => setHzn(parseInt(e.target.value, 10))} style={selectStyle()}>
              {[3, 6, 9, 12, 18, 24].map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              ))}
            </select>
          </label>
          <button onClick={loadFuture} style={btn()}>
            Generate Future
          </button>
        </div>

        {/* FUTURE chart */}
        <h3 style={{ ...h3(), marginTop: 10 }}>Future: Forecast</h3>
        {errFuture ? (
          <div style={{ color: "crimson" }}>Error: {errFuture}</div>
        ) : !future || !futureChart ? (
          <div>Loading…</div>
        ) : futureHasLine ? (
          <div style={chartBox()}>
            <Line data={futureChart.data} options={futureChart.options} />
          </div>
        ) : (
          <div
            style={{
              ...chartBox(),
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#666",
            }}
          >
            No future predictions yet. Try a different horizon.
          </div>
        )}

        {/* Meta */}
        <div style={{ marginTop: 12, color: "#333" }}>
          {past?.asof && (
            <span>
              <strong>Past as of:</strong> {new Date(past.asof).toLocaleString()}
            </span>
          )}
          {future?.asof && (
            <span style={{ marginLeft: 12 }}>
              <strong>Future as of:</strong> {new Date(future.asof).toLocaleString()}
            </span>
          )}
        </div>
      </div>
    </main>
  );
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
