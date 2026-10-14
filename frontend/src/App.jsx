
import { useState } from "react";
import "./App.css";

const pipelineSteps = [
  {
    number: "01",
    title: "Transaction input",
    description: "Capture customer, receiver, card, merchant, device and payment amount.",
  },
  {
    number: "02",
    title: "Feature extraction",
    description: "Prepare the 12 transaction features using available historical activity.",
  },
  {
    number: "03",
    title: "Sequence preparation",
    description: "Normalize features and arrange them into a 10-step model input.",
  },
  {
    number: "04",
    title: "BiLSTM processing",
    description: "Learn sequential transaction behaviour from the input history.",
  },
  {
    number: "05",
    title: "Transformer attention",
    description: "Learn important relationships and patterns within the sequence.",
  },
  {
    number: "06",
    title: "Risk decision",
    description: "Map the predicted probability to the configured payment decision.",
  },
];

function ShieldLogo() {
  return (
    <div className="brand-logo" aria-label="TransFraud shield logo">
      <svg viewBox="0 0 48 48" fill="none" aria-hidden="true">
        <path
          d="M24 4.5 39 10v12.2c0 9.2-6.1 16.3-15 21.3C15.1 38.5 9 31.4 9 22.2V10L24 4.5Z"
          stroke="currentColor"
          strokeWidth="2.8"
          strokeLinejoin="round"
        />
        <path
          d="m16.5 23.5 5 5 10-11"
          stroke="currentColor"
          strokeWidth="3.2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}

function App() {
  const [customerId, setCustomerId] = useState("CUST0000186");
  const [amount, setAmount] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [receiverId, setReceiverId] = useState("");
  const [cardId, setCardId] = useState("");
  const [merchantId, setMerchantId] = useState("");
  const [deviceId, setDeviceId] = useState("");

  const checkPayment = async () => {
    const trimmedCustomer = customerId.trim();
    const trimmedReceiver = receiverId.trim();
    const trimmedCard = cardId.trim();
    const trimmedMerchant = merchantId.trim();
    const trimmedDevice = deviceId.trim();
    const paymentAmount = Number(amount);

    if (
      !trimmedCustomer ||
      !trimmedReceiver ||
      !trimmedCard ||
      !trimmedMerchant ||
      !trimmedDevice
    ) {
      setError(
        "Please fill in all customer, receiver, card, merchant, and device fields."
      );
      setResult(null);
      return;
    }

    if (!amount || !Number.isFinite(paymentAmount) || paymentAmount <= 0) {
      setError("Please enter a valid payment amount greater than zero.");
      setResult(null);
      return;
    }

    setLoading(true);
    setResult(null);
    setError("");

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/predict-new-payment",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            customer_id: trimmedCustomer,
            receiver_id: trimmedReceiver,
            card_id: trimmedCard,
            merchant_id: trimmedMerchant,
            device_id: trimmedDevice,
            amount: paymentAmount,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || data.error || `Request failed (${response.status})`
        );
      }

      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the backend. Check that FastAPI is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const demoFraudCase = async () => {
    setLoading(true);
    setResult(null);
    setError("");

    try {
      const response = await fetch("http://127.0.0.1:8000/demo-fraud-case");
      const data = await response.json();

      if (!response.ok || data.error) {
        throw new Error(
          data.detail || data.error || `Request failed (${response.status})`
        );
      }

      setResult({
        customer_id: data.customer_id,
        receiver_id: "Demo Receiver",
        amount: data.amount,
        fraud_probability: data.fraud_probability,
        risk: data.risk,
        decision: data.decision,
        demo: true,
      });
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the backend. Check that FastAPI is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const probability = result
    ? Math.min(
        100,
        Math.max(0, Number(result.fraud_probability) * 100)
      )
    : null;

  const riskClass = result
    ? String(result.risk || "").toLowerCase()
    : "pending";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <ShieldLogo />
          <div className="brand-copy">
            <h1>TRANSFRAUD</h1>
            <p>
              A Real-Time Transformer-Based Model for Financial Fraud Detection
            </p>
          </div>
        </div>

        <div className="connection-status">
          <span className="status-dot" />
          Connected
        </div>
      </header>

      <main className="dashboard">
        <section className="panel form-panel">
          <div className="panel-heading">
            <h2>New assessment</h2>
            <p>Enter payment details</p>
          </div>

          <form
            className="payment-form"
            onSubmit={(event) => {
              event.preventDefault();
              checkPayment();
            }}
          >
            <div className="form-field">
              <label htmlFor="customerId">Customer ID</label>
              <input
                id="customerId"
                type="text"
                value={customerId}
                onChange={(e) => setCustomerId(e.target.value)}
                placeholder="Enter customer ID"
                autoComplete="off"
              />
            </div>

            <div className="form-field">
              <label htmlFor="receiverId">Receiver ID</label>
              <input
                id="receiverId"
                type="text"
                value={receiverId}
                onChange={(e) => setReceiverId(e.target.value)}
                placeholder="Enter receiver ID"
                autoComplete="off"
              />
            </div>

            <div className="form-field">
              <label htmlFor="cardId">Card ID</label>
              <input
                id="cardId"
                type="text"
                value={cardId}
                onChange={(e) => setCardId(e.target.value)}
                placeholder="Enter card ID"
                autoComplete="off"
              />
            </div>

            <div className="form-field">
              <label htmlFor="merchantId">Merchant ID</label>
              <input
                id="merchantId"
                type="text"
                value={merchantId}
                onChange={(e) => setMerchantId(e.target.value)}
                placeholder="Enter merchant ID"
                autoComplete="off"
              />
            </div>

            <div className="form-field">
              <label htmlFor="deviceId">Device ID</label>
              <input
                id="deviceId"
                type="text"
                value={deviceId}
                onChange={(e) => setDeviceId(e.target.value)}
                placeholder="Enter device ID"
                autoComplete="off"
              />
            </div>

            <div className="form-field">
              <label htmlFor="amount">Amount (₹)</label>
              <div className="amount-input">
                <span aria-hidden="true">₹</span>
                <input
                  id="amount"
                  type="number"
                  min="0.01"
                  step="any"
                  value={amount}
                  onChange={(e) => setAmount(e.target.value)}
                  placeholder="0.00"
                />
              </div>
            </div>

            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? "ASSESSING TRANSACTION..." : "ASSESS TRANSACTION"}
            </button>

            <button
              className="secondary-button"
              type="button"
              onClick={demoFraudCase}
              disabled={loading}
            >
              {loading ? "LOADING DEMO..." : "Run demo fraud case"}
            </button>
          </form>

          {error && (
            <div className="error-box" role="alert">
              {error}
            </div>
          )}

          <p className="form-footnote">
            Risk assessment prototype · No real payment is processed.
          </p>
        </section>

        <section className="panel assessment-panel">
          <div className="assessment-heading">
            <span className="eyebrow">RISK ASSESSMENT</span>
            <h2>{loading ? "Analyzing transaction" : result ? "Assessment complete" : "Awaiting transaction"}</h2>
            {result?.demo && (
              <span className="demo-label">DEMO · HISTORICAL TEST CASE</span>
            )}
          </div>

          <div className="probability-card">
            <div className="probability-top">
              <span>Fraud probability</span>
              <strong className={result ? `score ${riskClass}` : "score"}>
                {result ? `${probability.toFixed(2)}%` : "— %"}
              </strong>
            </div>

            <div
              className="probability-track"
              role="progressbar"
              aria-label="Fraud probability"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={probability ?? 0}
            >
              <div
                className={`probability-fill ${riskClass}`}
                style={{ width: `${probability ?? 0}%` }}
              />
            </div>

            <p>
              {result
                ? "Model-estimated fraud score for this transaction."
                : "Run an assessment to see the model score."}
            </p>
          </div>

          <div className="summary-grid">
            <div className="summary-card">
              <span>Risk level</span>
              <strong className={result ? `risk-value ${riskClass}` : ""}>
                {result?.risk || "Pending"}
              </strong>
            </div>
            <div className="summary-card">
              <span>Decision</span>
              <strong>{result?.decision || "Pending"}</strong>
            </div>
          </div>

          {result && (
            <div className="transaction-summary">
              <div>
                <span>Customer</span>
                <strong>{result.customer_id}</strong>
              </div>
              <div>
                <span>Receiver</span>
                <strong>{result.receiver_id}</strong>
              </div>
              <div>
                <span>Amount</span>
                <strong>₹ {Number(result.amount).toLocaleString("en-IN")}</strong>
              </div>
            </div>
          )}

          <div className="pipeline-section">
            <div className="pipeline-heading">
              <h3>Detection pipeline</h3>
              <span>06 STEPS</span>
            </div>

            <div className="pipeline-list">
              {pipelineSteps.map((step) => (
                <div className="pipeline-step" key={step.number}>
                  <div className="step-number">{step.number}</div>
                  <div className="step-copy">
                    <strong>{step.title}</strong>
                    <p>{step.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <span>TRANSFRAUD · Financial transaction risk analysis</span>
        <span>BiLSTM + Transformer</span>
      </footer>
    </div>
  );
}

export default App;
