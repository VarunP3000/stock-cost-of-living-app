// frontend/lib/api.ts
import {
  CountriesResponse as CountriesSchema,
  SeriesResponse as SeriesSchema,
  CorrelationsResponse as CorrelationsSchema,
  ForecastResponse as ForecastSchema,
  ArtifactsResponse as ArtifactsSchema,
} from "@/types/api";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://127.0.0.1:8000";

// ---------- core fetch helpers ----------

/** Raw fetch (typed as unknown; caller validates with zod) */
async function getRaw(path: string): Promise<unknown> {
  const url = `${API_BASE}${path}`;
  const r = await fetch(url, { headers: { Accept: "application/json" } });
  if (!r.ok) throw new Error(`GET ${url} -> ${r.status} ${r.statusText}`);
  return r.json();
}

/** Fetch + Zod-validate */
async function get<T>(path: string, schema: { parse: (d: unknown) => T }): Promise<T> {
  const data = await getRaw(path);
  return schema.parse(data);
}

// ---------- API ----------

export const api = {
  countries: () => get("/countries", CountriesSchema),

  /** Resilient series: if backend returns a bare array, wrap it to match schema */
  series: async (geo: string, kind: string) => {
    const path = `/series?geo=${encodeURIComponent(geo)}&kind=${encodeURIComponent(kind)}`;
    const data = await getRaw(path);
    const wrapped = Array.isArray(data) ? { geo, kind, series: data } : data;
    return SeriesSchema.parse(wrapped);
  },

  correlations: (windowMonths: number, geo?: string) => {
    const params = new URLSearchParams({ window_months: String(windowMonths) });
    if (geo) params.set("geo", geo);
    return get(`/correlations?${params.toString()}`, CorrelationsSchema);
  },

  forecastModel: (model: "ridge" | "elasticnet" | "gb" | "quantiles" | "directional") => {
    if (model === "quantiles") return getRaw(`/forecast/quantiles`); // custom shape
    return get(`/forecast/${model}`, ForecastSchema);
  },

  forecastEnsemble: (
    weights?: Record<string, number>,
    opts?: { scenario?: "baseline" | "optimistic" | "pessimistic"; horizon?: number }
  ) => {
    const params = new URLSearchParams();
    if (weights && Object.keys(weights).length) {
      const w = Object.entries(weights)
        .filter(([, v]) => typeof v === "number" && v > 0)
        .map(([k, v]) => `${k}:${v}`)
        .join(",");
      if (w) params.set("weights", w);
    }
    if (opts?.scenario) params.set("scenario", opts.scenario);
    if (opts?.horizon) params.set("horizon", String(opts.horizon));
    const qs = params.toString() ? `?${params.toString()}` : "";
    return getRaw(`/forecast/ensemble${qs}`);
  },

  quantilesPoint: (horizon = 1, scenario: "baseline" | "optimistic" | "pessimistic" = "baseline") =>
    getRaw(`/forecast/quantiles?horizon=${horizon}&scenario=${scenario}`),

  forecastRegional: (region: "americas" | "emea" | "apac") =>
    get(`/forecast/regional?region=${region}`, ForecastSchema),

  artifacts: () => get("/artifacts", ArtifactsSchema),
};

// ----------------- Metrics helpers -----------------

export function formatNumber(n: number | undefined, opts: Intl.NumberFormatOptions = {}) {
  if (typeof n !== "number" || Number.isNaN(n)) return "—";
  return new Intl.NumberFormat(undefined, opts).format(n);
}

export function formatPct(p: number | undefined, digits = 2) {
  if (typeof p !== "number" || Number.isNaN(p)) return "—";
  return `${(p * 100).toFixed(digits)}%`;
}

/** Thin facade tailored for the homepage MetricsPanel */
export const metricsApi = {
  ensemble: (weights?: Record<string, number>) =>
    api.forecastEnsemble(weights, { scenario: "baseline", horizon: 1 }),
  correlationUS36m: () => api.correlations(36, "United States"),
  quantilesPoint: () => api.quantilesPoint(1, "baseline") as Promise<unknown>,
  quantiles: () => api.quantilesPoint(1, "baseline") as Promise<unknown>,
};

export type {
  CountriesResponse,
  SeriesResponse,
  CorrelationsResponse,
  ForecastResponse,
  ArtifactsResponse,
} from "@/types/api";
