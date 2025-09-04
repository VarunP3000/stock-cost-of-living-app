// frontend/lib/api.ts
import {
  CountriesResponse as CountriesSchema,
  SeriesResponse as SeriesSchema,
  CorrelationsResponse as CorrelationsSchema,
  ForecastResponse as ForecastSchema,
  ArtifactsResponse as ArtifactsSchema,
} from "@/types/api";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";

/** Raw fetch (no validation) */
async function getRaw(path: string): Promise<any> {
  const url = `${API_BASE}${path}`;
  const r = await fetch(url, { headers: { Accept: "application/json" } });
  if (!r.ok) throw new Error(`GET ${url} -> ${r.status}`);
  return r.json();
}

/** Fetch + Zod-validate */
async function get<T>(path: string, schema: { parse: (d: unknown) => T }): Promise<T> {
  const data = await getRaw(path);
  return schema.parse(data);
}

export const api = {
  countries: () => get("/countries", CountriesSchema),

  /** Resilient series: if backend returns a bare array, wrap it to match schema */
  series: async (geo: string, kind: string) => {
    const path = `/series?geo=${encodeURIComponent(geo)}&kind=${encodeURIComponent(kind)}`;
    const data = await getRaw(path);
    const wrapped = Array.isArray(data) ? { geo, kind, series: data } : data;
    return SeriesSchema.parse(wrapped);
  },

  correlations: (windowMonths: number) =>
    get(`/correlations?window_months=${windowMonths}`, CorrelationsSchema),

  // Forecast endpoints
  forecastModel: (model: "ridge" | "elasticnet" | "gb" | "quantiles" | "directional") =>
    get(`/forecast/${model}`, ForecastSchema),

  forecastEnsemble: (weights?: Record<string, number>) => {
    const qs =
      weights && Object.keys(weights).length
        ? `?${new URLSearchParams({
            weights: Object.entries(weights)
              .map(([k, v]) => `${k}:${v}`)
              .join(","),
          }).toString()}`
        : "";
    return get(`/forecast/ensemble${qs}`, ForecastSchema);
  },

  forecastRegional: (region: "americas" | "emea" | "apac") =>
    get(`/forecast/regional?region=${region}`, ForecastSchema),

  artifacts: () => get("/artifacts", ArtifactsSchema),
};

export type {
  CountriesResponse,
  SeriesResponse,
  CorrelationsResponse,
  ForecastResponse,
  ArtifactsResponse,
} from "@/types/api";
