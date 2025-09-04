// frontend/types/forecast.ts
export type WeightMap = Record<string, number>;

export type PastPoint = {
  asof: string;
  ensemble: number;
  actual: number;
  model_std: number;
};

export type MetricMap = {
  rmse: number;
  mae: number;
  mape: number;
  hit_rate: number;
};

export type PastPayload = {
  points: PastPoint[];
  metrics: MetricMap;
  model_weights: WeightMap;
  model_list: string[];
  model_panel: Array<Record<string, number | string | null>>;
};

export type FuturePoint = {
  asof: string;
  yhat: number;
  disagreement: number;
};

export type FutureBand = {
  asof: string;
  yhat_lo: number;
  yhat_hi: number;
};

export type FuturePayload = {
  forecast: FuturePoint[];
  confidence_band: FutureBand[];
  model_weights: WeightMap;
  model_list: string[];
};
