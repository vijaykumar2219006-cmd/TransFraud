import { useState } from "react";
import "./App.css";

function App() {
  const [customerId, setCustomerId] = useState("CUST0000186");
  const [amount, setAmount] = useState("");
  const [newDevice, setNewDevice] = useState(false);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [receiverId, setReceiverId] = useState("");

  const checkPayment = async () => {
    setLoading(true);
    setResult(null);
    setError("");

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/predict-new-payment",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            customer_id: customerId,
            receiver_id: receiverId,
            amount: Number(amount),
            new_device: newDevice ? 1 : 0,
          }),
        },
      );

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>TransFraud</h1>
        <p>AI-Powered Pre-Payment Fraud Detection</p>
      </header>

      <main>
        <div className="demo-notice">
          Demo Mode — Transaction risk is assessed before payment authorization.
        </div>

        <div className="payment-card">
          <h2>Secure Payment</h2>
          <p>Enter the payment details to check fraud risk.</p>

          <label>Customer ID</label>
          <input
            type="text"
            value={customerId}
            onChange={(e) => setCustomerId(e.target.value)}
            placeholder="CUST0000186"
          />

          <label>Receiver ID</label>
          <input
            type="text"
            value={receiverId}
            onChange={(e) => setReceiverId(e.target.value)}
            placeholder="Enter receiver ID"
          />

          <label>Amount</label>
          <input
            type="number"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
            placeholder="Enter amount"
          />

          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={newDevice}
              onChange={(e) => setNewDevice(e.target.checked)}
            />
            Payment from a new device
          </label>

          <button onClick={checkPayment} disabled={loading || !amount}>
            {loading ? "CHECKING TRANSACTION..." : "CHECK & MAKE PAYMENT"}
          </button>
        </div>

        {error && <div className="error-box">{error}</div>}

        {result && (
          <div className="result-card">
            <div className="result-header">
              <h2>TransFraud Risk Assessment</h2>
              <span>✓ Checked Before Payment</span>
            </div>

            <div className="result-grid">
              <div>
                <strong>Customer</strong>
                <p>{result.customer_id}</p>
              </div>

              <div>
                <strong>Receiver</strong>
                <p>{result.receiver_id}</p>
              </div>

              <div>
                <strong>Amount</strong>
                <p>{result.amount}</p>
              </div>

              <div>
                <strong>Fraud Probability</strong>
                <p>{(result.fraud_probability * 100).toFixed(2)}%</p>
              </div>

              <div>
                <strong>Risk</strong>
                <p className={`risk ${result.risk.toLowerCase()}`}>
                  {result.risk}
                </p>
              </div>
            </div>

            <div className="decision">
              <strong>Payment Decision</strong>
              <p>{result.decision}</p>
            </div>
          </div>
        )}

        <div className="pipeline">
          <h3>Detection Pipeline</h3>

          <div className="pipeline-items">
            <span>Payment</span>→<span>History</span>→
            <span>10×12 Sequence</span>→<span>BiLSTM</span>→
            <span>Transformer</span>→<span>Risk</span>→<span>Decision</span>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
