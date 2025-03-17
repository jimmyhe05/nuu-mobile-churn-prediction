import React, { useState } from "react";
import { Container, Row, Col, Card, Table } from "react-bootstrap";
import { Link, useLocation, useParams } from "react-router-dom";
import { Pie } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";

// Register the components required for the Pie chart
ChartJS.register(ArcElement, Tooltip, Legend);

export default function Predictions() {
  const location = useLocation();
  const { fileName } = useParams(); // Extract fileName from the URL
  const { predictions } = location.state;

  // State for risk threshold (default: 50%)
  const [riskThreshold, setRiskThreshold] = useState(50);

  // Filter risky customers based on the current threshold
  const riskyCustomers = predictions.filter(
    (p) => p.churn_probability > riskThreshold / 100
  );

  // Data for the pie chart
  const pieData = {
    labels: ["Churn", "Non-Churn"],
    datasets: [
      {
        data: [
          predictions.filter((p) => p.churn_probability > riskThreshold / 100)
            .length,
          predictions.filter((p) => p.churn_probability <= riskThreshold / 100)
            .length,
        ],
        backgroundColor: ["#ff4d4d", "#0088FE"],
      },
    ],
  };

  // Handle slider and input changes
  const handleThresholdChange = (e) => {
    const value = Math.min(100, Math.max(0, e.target.value)); // Ensure value is between 0 and 100
    setRiskThreshold(value);
  };

  return (
    <Container fluid className="mt-4">
      <Link to="/" className="btn btn-secondary mb-4">
        <i className="bi bi-arrow-left"></i> Back to Dashboard
      </Link>
      <h1 className="text-center mb-4">Prediction Results</h1>
      <Row>
        <Col md={4}>
          <Card className="shadow p-3">
            {/* File Name */}
            <div className="d-flex align-items-center mb-3">
              <i className="bi bi-file-earmark-text fs-4 me-2"></i>{" "}
              {/* Bootstrap Icon */}
              <h5 className="mb-0">File Name: {fileName}</h5>
            </div>

            {/* Risky Customers */}
            <div className="d-flex align-items-center mb-3">
              <i className="bi bi-exclamation-triangle-fill text-danger fs-4 me-2"></i>{" "}
              {/* Bootstrap Icon */}
              <h5 className="mb-0 text-danger">
                Risky Customers:{" "}
                <span className="badge bg-danger">{riskyCustomers.length}</span>
              </h5>
            </div>

            {/* Risk Threshold Slider and Input */}
            <div className="mb-3">
              <label htmlFor="riskThreshold" className="form-label">
                Risk Threshold: {riskThreshold}%
              </label>
              <input
                type="range"
                className="form-range"
                id="riskThreshold"
                min="0"
                max="100"
                value={riskThreshold}
                onChange={handleThresholdChange}
              />
              <input
                type="number"
                className="form-control mt-2"
                min="0"
                max="100"
                value={riskThreshold}
                onChange={handleThresholdChange}
              />
            </div>

            {/* Total Customers */}
            <div className="d-flex align-items-center mb-4">
              <i className="bi bi-people-fill fs-4 me-2"></i>{" "}
              {/* Bootstrap Icon */}
              <h5 className="mb-0">
                Total Customers:{" "}
                <span className="badge bg-primary">{predictions.length}</span>
              </h5>
            </div>

            {/* Pie Chart */}
            <div className="text-center">
              <Pie data={pieData} />
            </div>
          </Card>
        </Col>
        <Col md={8}>
          <Card className="shadow p-3">
            <Table responsive>
              <thead>
                <tr>
                  <th>Customer Number</th>
                  <th>Device Number</th>
                  <th>Chance of Churn</th>
                </tr>
              </thead>
              <tbody>
                {predictions
                  .sort((a, b) => b.churn_probability - a.churn_probability)
                  .map((prediction, index) => (
                    <tr key={index}>
                      {/* Customer Number */}
                      <td
                        className={
                          prediction.churn_probability > riskThreshold / 100
                            ? "text-danger"
                            : ""
                        }
                      >
                        {prediction.customer_number}
                      </td>
                      {/* Device Number */}
                      <td
                        className={
                          prediction.churn_probability > riskThreshold / 100
                            ? "text-danger"
                            : ""
                        }
                      >
                        {prediction["device number"]}{" "}
                        {/* Access device number with bracket notation */}
                      </td>
                      {/* Chance of Churn */}
                      <td
                        className={
                          prediction.churn_probability > riskThreshold / 100
                            ? "text-danger"
                            : ""
                        }
                      >
                        {(prediction.churn_probability * 100).toFixed(2)}%
                      </td>
                    </tr>
                  ))}
              </tbody>
            </Table>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}
