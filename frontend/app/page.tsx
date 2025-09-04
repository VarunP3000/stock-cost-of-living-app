// frontend/app/page.tsx
import Link from "next/link";
import { api } from "@/lib/api";
import MetricsPanel from "./components/MetricsPanel";

export default async function HomePage() {
  // Server-side check for live coverage
  let countriesCount = 0;
  try {
    const countries = await api.countries();
    countriesCount = countries.length;
  } catch {
    countriesCount = 0;
  }

  return (
    <main style={styles.main}>
      {/* Hero */}
      <section style={styles.hero}>
        <div style={styles.heroText}>
          <h1 style={styles.h1}>Understand how inflation impacts the stock market</h1>
          <p style={styles.sub}>
            An interactive platform connecting global CPI trends with S&amp;P 500 performance.
            Explore data, correlations, and ML forecasts—then generate a report in one click.
          </p>
          <div style={styles.ctaRow}>
            <Link href="/dashboard" style={{ ...styles.btn, ...styles.btnPrimary }}>
              Explore Dashboard
            </Link>
            <Link href="/forecast" style={{ ...styles.btn, ...styles.btnGhost }}>
              Run Forecast
            </Link>
            <Link href="/reports" style={{ ...styles.btn, ...styles.btnGhost }}>
              Download Report
            </Link>
          </div>
          <p style={styles.meta}>
            Live data coverage: <strong>{countriesCount}</strong> countries
          </p>
        </div>

        {/* 🔥 Replace static KPI grid with live MetricsPanel */}
        <div style={styles.heroCard}>
          <MetricsPanel />
        </div>
      </section>

      {/* Feature rows */}
      <section style={styles.features}>
        <h2 style={styles.h2}>What you can do</h2>
        <div style={styles.cardGrid}>
          <article style={styles.card}>
            <h3 style={styles.cardTitle}>Visualize CPI vs SPX</h3>
            <p style={styles.cardBody}>
              Select any country and compare CPI vs SPX returns over time. Zoom, pan, and change
              windows for rolling correlations.
            </p>
            <Link href="/dashboard" style={styles.cardLink}>
              Go to Dashboard →
            </Link>
          </article>

          <article style={styles.card}>
            <h3 style={styles.cardTitle}>Run ML Forecasts</h3>
            <p style={styles.cardBody}>
              Generate next-month SPX predictions using Ridge, ElasticNet, Gradient Boosting, or an
              Ensemble—complete with confidence bands.
            </p>
            <Link href="/forecast" style={styles.cardLink}>
              Run a Forecast →
            </Link>
          </article>

          <article style={styles.card}>
            <h3 style={styles.cardTitle}>Export Reports</h3>
            <p style={styles.cardBody}>
              Produce a PDF/HTML report summarizing CPI→SPX trends, correlations, and your latest
              forecasts for easy sharing.
            </p>
            <Link href="/reports" style={styles.cardLink}>
              Generate Report →
            </Link>
          </article>
        </div>
      </section>

      {/* Developer/API section */}
      <section style={styles.api}>
        <h2 style={styles.h2}>For developers & researchers</h2>
        <p style={styles.cardBody}>
          Full API access with Swagger docs. Endpoints include: <code>/countries</code>,{" "}
          <code>/series</code>, <code>/correlations</code>, <code>/forecast</code>,{" "}
          <code>/artifacts</code>.
        </p>
        <div style={styles.ctaRow}>
          <a
            href={`${process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000"}/docs`}
            target="_blank"
            rel="noreferrer"
            style={{ ...styles.btn, ...styles.btnGhost }}
          >
            Open API Docs
          </a>
        </div>
      </section>
    </main>
  );
}

const styles: Record<string, React.CSSProperties> = {
  main: {
    padding: "28px 20px 60px",
    maxWidth: 1100,
    margin: "0 auto",
    background: "black", // ✅ dark background
    color: "white",      // ✅ white text
  },
  hero: {
    display: "grid",
    gridTemplateColumns: "1.1fr 0.9fr",
    gap: 24,
    alignItems: "stretch",
    marginBottom: 36,
  },
  heroText: { display: "flex", flexDirection: "column", gap: 12 },
  h1: { fontSize: 34, lineHeight: 1.2, margin: 0, fontWeight: 700, color: "white" },
  sub: { fontSize: 16, color: "#ccc", margin: 0 },
  ctaRow: { display: "flex", gap: 12, marginTop: 12, flexWrap: "wrap" },
  btn: {
    padding: "10px 14px",
    borderRadius: 10,
    textDecoration: "none",
    fontWeight: 600,
    border: "1px solid #555",
  },
  btnPrimary: { background: "white", color: "black" },
  btnGhost: { background: "transparent", color: "white", border: "1px solid white" },
  meta: { fontSize: 13, opacity: 0.8, marginTop: 10 },
  heroCard: {
    border: "1px solid #333",
    borderRadius: 14,
    padding: 16,
    background: "#111",
    overflow: "hidden",        // <- add this
  },
  features: { marginTop: 10 },
  h2: { fontSize: 22, margin: "4px 0 12px", fontWeight: 700, color: "white" },
  cardGrid: { display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 },
  card: { border: "1px solid #333", borderRadius: 14, padding: 16, background: "#111" },
  cardTitle: { fontSize: 16, margin: "0 0 8px", fontWeight: 700, color: "white" },
  cardBody: { fontSize: 14, margin: 0, color: "#ccc" },
  cardLink: {
    display: "inline-block",
    marginTop: 10,
    fontWeight: 600,
    textDecoration: "none",
    color: "#0af",
  },
  api: { marginTop: 36, borderTop: "1px solid #333", paddingTop: 24 },
};

