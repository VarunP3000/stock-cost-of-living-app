// frontend/types/api.ts
import { z } from "zod";

/** Shared **/
export const ISODate = z.string().regex(/^\d{4}-\d{2}-\d{2}/, "ISO date (YYYY-MM-DD)");

export const Metrics = z.object({
  rmse: z.number().nullable().optional(),
  mae: z.number().nullable().optional(),
  r2: z.number().nullable().optional(),
  directional_accuracy: z.number().nullable().optional(),
  samples: z.number().nullable().optional(),
  trained_on: z.string().nullable().optional(),
  evaluated_to: z.string().nullable().optional(),
  updated_at: z.string().nullable().optional(),
});

/** /countries **/
export const CountriesResponse = z.array(z.string());
export type CountriesResponse = z.infer<typeof CountriesResponse>;

/** /series?geo=...&kind=cpi_yoy|spx_ret **/
export const SeriesPoint = z.tuple([ISODate, z.number().nullable()]);
export const SeriesResponse = z.object({
  geo: z.string(),
  kind: z.string(),
  series: z.array(SeriesPoint),
});
export type SeriesResponse = z.infer<typeof SeriesResponse>;

/** /correlations?window_months=36 **/
export const CorrelationsResponse = z.object({
  window_months: z.number(),
  countries: z.array(z.string()),   // column labels
  dates: z.array(ISODate),          // row labels
  matrix: z.array(z.array(z.number().nullable())), // shape: dates x countries
});
export type CorrelationsResponse = z.infer<typeof CorrelationsResponse>;

/** /forecast/... **/
export const ForecastMetadata = z.object({
  artifact: z.string().optional(),
  region: z.string().optional(),
  metrics: Metrics.optional(),
  components: z
    .array(
      z.object({
        route: z.string(),
        prediction: z.number(),
      })
    )
    .optional(),
  // zod@v3 requires both key and value types:
  weights_used: z.record(z.string(), z.number()).optional(),
});

export const ForecastResponse = z.object({
  asof: z.string(),
  prediction: z.number(),
  feature_order: z.union([z.string(), z.array(z.string())]).optional(),
  metadata: ForecastMetadata.optional(),
  // quantile regressions
  p10: z.number().optional(),
  p50: z.number().optional(),
  p90: z.number().optional(),
  // directional classifier
  up_probability: z.number().min(0).max(1).optional(),
});
export type ForecastResponse = z.infer<typeof ForecastResponse>;

/** /artifacts **/
export const ArtifactRow = z.object({
  name: z.string(),
  region: z.string().optional(),
  metrics: Metrics.optional(),
  created_at: z.string().optional(),
});
export const ArtifactsResponse = z.array(ArtifactRow);
export type ArtifactsResponse = z.infer<typeof ArtifactsResponse>;
