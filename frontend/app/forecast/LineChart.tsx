// frontend/app/forecast/LineChart.tsx
"use client";

import {
  LineChart as RCLineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";

type Series = {
  name: string;
  data: { x: string; y: number | null }[];
};

export default function LineChart({ series }: { series: Series[] }) {
  // merge by x for a simple multi-line chart
  const xKeys = Array.from(new Set(series.flatMap((s) => s.data.map((d) => d.x)))).sort();
  const rows = xKeys.map((x) => {
    const row: Record<string, string | number | null> = { x };
    for (const s of series) {
      const hit = s.data.find((d) => d.x === x);
      row[s.name] = hit?.y ?? null;
    }
    return row;
  });

  return (
    <div style={{ width: "100%", height: 380 }}>
      <ResponsiveContainer>
        <RCLineChart data={rows} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="x" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip />
          <Legend />
          {series.map((s) => (
            <Line
              key={s.name}
              type="monotone"
              dataKey={s.name}
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </RCLineChart>
      </ResponsiveContainer>
    </div>
  );
}
