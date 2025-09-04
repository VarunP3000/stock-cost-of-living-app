// frontend/app/dashboard/page.tsx
import Link from "next/link";
import { api } from "@/lib/api";
import DashboardClient from "./DashboardClient";

export default async function DashboardPage() {
  // Fetch countries on the server once
  let countries: string[] = [];
  try {
    countries = await api.countries();
  } catch {
    countries = ["United States", "Canada", "United Kingdom"]; // fallback
  }

  const initial = countries[0] ?? "United States";

  return (
    <main
      style={{
        minHeight: "100vh",
        background: "#0b0b0b", // page backdrop (dark)
        padding: 24,
      }}
    >
      <div
        style={{
          maxWidth: 1100,
          margin: "0 auto",
          background: "#ffffff", // main content card (light)
          color: "#111111",
          borderRadius: 16,
          border: "1px solid #e6e6e6",
          padding: 20,
          boxShadow: "0 8px 24px rgba(0,0,0,0.1)",
        }}
      >
        {/* Back button */}
        <div style={{ marginBottom: 12 }}>
          <Link
            href="/"
            style={{
              display: "inline-block",
              padding: "8px 12px",
              borderRadius: 10,
              border: "1px solid #d0d0d0",
              textDecoration: "none",
              background: "#f8f8f8",
              color: "#111",
              fontWeight: 600,
            }}
          >
            ← Back
          </Link>
        </div>

        <h1 style={{ fontSize: 28, margin: "4px 0 4px", fontWeight: 800, color: "#111" }}>
          Dashboard
        </h1>
        <p style={{ margin: "0 0 16px", color: "#333" }}>
          Explore CPI vs SPX and rolling correlations. Pick a country to begin.
        </p>

        <DashboardClient initialCountry={initial} countries={countries} />
      </div>
    </main>
  );
}
