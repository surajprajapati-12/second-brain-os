
import { useEffect, useState } from "react";
import "./App.css";

type HealthResponse = {
  status: string;
  service: string;
};

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const apiUrl = import.meta.env.VITE_API_URL;

    fetch(`${apiUrl}/health`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Backend request failed");
        }
        return response.json();
      })
      .then((data: HealthResponse) => setHealth(data))
      .catch(() => setError("Could not connect to the backend."));
  }, []);

  return (
    <main className="app">
      <section className="panel">
        <p className="eyebrow">YOUR PERSONAL KNOWLEDGE SYSTEM</p>
        <h1>Second Brain OS</h1>
        <p className="subtitle">
          Capture your knowledge. Connect your ideas. Find answers faster.
        </p>

        <div className="status">
          <span
            className={`status-dot ${health ? "online" : error ? "offline" : ""}`}
          />
          <strong>
            {health ? "Backend connected" : error || "Connecting to backend..."}
          </strong>
        </div>

        {health && (
          <p className="details">
            Service: {health.service} · Status: {health.status}
          </p>
        )}
      </section>
    </main>
  );
}

export default App;
