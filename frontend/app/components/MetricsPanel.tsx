// frontend/app/components/MetricsPanel.tsx
"use client";

import { useEffect, useState } from "react";
import { metricsApi, formatNumber, formatPct } from "@/lib/api";

type State =
  | { status: "loading" }
  | { status: "ready"; ensemble?: any; corr?: any; q?: any }
  | { status: "error"; message: string };

export default function MetricsPanel() {
  const [state, setState] = useState<State>({ status: "loading" });

  async function load() {
    try {
      const [ensemble, corr, q] = await Promise.all([
        metricsApi.ensemble().catch(() => undefined),
        metricsApi.correlationUS36m().catch(() => undefined),
        metricsApi.quantiles().catch(() => undefined),
      ]);
      setState({ status: "ready", ensemble, corr, q });
    } catch (err: any) {
      setState({ status: "error", message: err?.message ?? "Failed to load metrics" });
    }
  }

  useEffect(() => {
    load();
    const id = setInterval(load, 60_000);
    return () => clearInterval(id);
  }, []);

  if (state.status === "loading") return <SkeletonGrid />;

  if (state.status === "error") {
    return (
      <div style={styles.error}>
        Couldn’t load metrics: {state.message}
      </div>
    );
  }

  const ensValue =
    typeof state.ensemble?.prediction === "number"
      ? formatNumber(state.ensemble.prediction, { maximumFractionDigits: 3 })
      : "—";

  const corrValue =
    typeof state.corr?.corr === "number" ? formatPct(state.corr.corr, 1) : "—";

  const spread =
    typeof state.q?.p10 === "number" && typeof state.q?.p90 === "number"
      ? `${formatNumber(state.q.p10, { maximumFractionDigits: 3 })} to ${formatNumber(
          state.q.p90,
          { maximumFractionDigits: 3 }
        )}`
      : "—";

  return (
    <div style={styles.grid}>
      {Card("Latest Forecast (Ensemble)", ensValue, "GET /forecast/ensemble")}
      {Card("3y CPI↔SPX Corr (US)", corrValue, "GET /correlations")}
      {Card("Quantile Band (p10–p90)", spread, "GET /forecast/quantiles")}
    </div>
  );
}

function Card(title: string, value: string, hint?: string) {
  return (
    <div style={styles.card}>
      <div style={styles.label}>{title}</div>
      <div style={styles.value}>{value}</div>
      {hint && <div style={styles.hint}>{hint}</div>}
    </div>
  );
}

function SkeletonGrid() {
  return (
    <div style={styles.grid}>
      <div style={styles.skelCard}>
        <div style={styles.skelLineSm} />
        <div style={styles.skelLineLg} />
        <div style={styles.skelLineXs} />
      </div>
      <div style={styles.skelCard}>
        <div style={styles.skelLineSm} />
        <div style={styles.skelLineLg} />
        <div style={styles.skelLineXs} />
      </div>
      <div style={styles.skelCard}>
        <div style={styles.skelLineSm} />
        <div style={styles.skelLineLg} />
        <div style={styles.skelLineXs} />
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  grid: {
    display: "grid",
    gap: 12,
    gridTemplateColumns: "repeat(3, 1fr)",
  },
  card: {
    border: "1px solid #333",
    borderRadius: 12,
    padding: 12,
    background: "#000",
  },
  label: { fontSize: 12, color: "#ccc" },
  value: { fontSize: 22, fontWeight: 700, marginTop: 4, color: "white" },
  hint: { fontSize: 12, color: "#9aa", marginTop: 2 },

  // skeletons
  skelCard: {
    border: "1px solid #333",
    borderRadius: 12,
    padding: 12,
    background: "#0b0b0b",
  },
  skelLineSm: { height: 10, width: 110, background: "#1f1f1f", borderRadius: 6 },
  skelLineLg: {
    height: 22,
    width: 80,
    background: "#1f1f1f",
    borderRadius: 6,
    marginTop: 10,
  },
  skelLineXs: { height: 10, width: 160, background: "#1f1f1f", borderRadius: 6, marginTop: 8 },

  error: {
    border: "1px solid #5b1f1f",
    borderRadius: 12,
    padding: 12,
    background: "#2a0f0f",
    color: "#f6bdbd",
    fontSize: 14,
  },
};
