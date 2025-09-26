// frontend/app/components/MetricsPanel.tsx
"use client";

import { useEffect, useState } from "react";
import { metricsApi, formatNumber, formatPct } from "@/lib/api";
import type { CorrelationsResponse } from "@/types/api";

/** Minimal shape we actually display from /forecast/ensemble */
type EnsemblePoint = {
  prediction?: number;
  model?: string;
  asof?: string;
  horizon?: number;
  scenario?: string;
};

/** Quantile point as normalized by metricsApi.quantilesPoint() */
type QuantilesPoint = {
  p10?: number;
  p50?: number;
  p90?: number;
} | null | undefined;

type ReadyState = {
  status: "ready";
  ensemble?: EnsemblePoint;
  corr?: CorrelationsResponse;
  q?: QuantilesPoint;
};

type State =
  | { status: "loading" }
  | ReadyState
  | { status: "error"; message: string };

export default function MetricsPanel() {
  const [state, setState] = useState<State>({ status: "loading" });

  async function load() {
    try {
      const [ensemble, corr, q] = await Promise.all([
        metricsApi.ensemble().catch(() => undefined) as Promise<EnsemblePoint | undefined>,
        metricsApi.correlationUS36m().catch(() => undefined) as Promise<CorrelationsResponse | undefined>,
        metricsApi.quantilesPoint().catch(() => undefined) as Promise<QuantilesPoint>,
      ]);
      setState({ status: "ready", ensemble, corr, q });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to load metrics";
      setState({ status: "error", message: msg });
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

  // Ensemble value
  const ensValue =
    typeof state.ensemble?.prediction === "number"
      ? formatNumber(state.ensemble.prediction, { maximumFractionDigits: 3 })
      : "—";

  // Latest US correlation from full matrix (36m window)
  const corrValue = (() => {
    const c = state.corr;
    if (!c) return "—";
    const usIdx = c.countries.findIndex((x) => x === "United States");
    const lastRow = c.matrix.at(-1);
    const v = usIdx >= 0 && lastRow ? lastRow[usIdx] : null;
    return typeof v === "number" && Number.isFinite(v) ? formatPct(v, 1) : "—";
  })();

  // Quantile spread (p10–p90)
  const spread = (() => {
    const q = state.q || undefined;
    const p10 = typeof q?.p10 === "number" ? q.p10 : undefined;
    const p90 = typeof q?.p90 === "number" ? q.p90 : undefined;
    if (p10 === undefined || p90 === undefined) return "—";
    return `${formatNumber(p10, { maximumFractionDigits: 3 })} to ${formatNumber(p90, {
      maximumFractionDigits: 3,
    })}`;
  })();

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
