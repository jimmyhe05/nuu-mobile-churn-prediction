import {
  Container,
  Row,
  Col,
  Card,
  Button,
  Table,
  Modal,
  Spinner,
} from "react-bootstrap";
import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  LineChart,
  Line,
  ResponsiveContainer,
} from "recharts";
import { FaTimes } from "react-icons/fa"; // Import the X icon

export default function Dashboard() {
  const [predictions, setPredictions] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const navigate = useNavigate();
  const [featureImportance, setFeatureImportance] = useState([]);

  // Fetch risk threshold from local storage (default: 50%)
  const [riskThreshold, setRiskThreshold] = useState(
    parseFloat(localStorage.getItem("riskThreshold")) || 50
  );

  // Helper function to calculate risky customers
  const calculateRiskyCustomers = (predictions, threshold) => {
    return predictions.filter((p) => p.churn_probability > threshold / 100).length;
  };

  // Helper function to format feature names
  const formatFeatureName = (feature) => {
    const replacements = {
      num__sim_info: "Number of Sim",
      num__days_since_activation: "Days Since Activation",
      num__days_since_last_use: "Days Since Last Use",
      num__days_used_since_activation: "Days Used Since Activation",
      num__product_model_encoded: "Product Model",
      num__promotion_email: "Promotion Email",
      num__register_email: "Registered Email",
    };

    return replacements[feature] || feature;
  };

  // Fetch Dashboard Data
  useEffect(() => {
    // Fetch Dashboard Data
    fetch("http://127.0.0.1:5000/dashboard_data")
      .then((response) => response.json())
      .then((data) => {
        data.app_usage_percentages.sort((a, b) => b.percentage - a.percentage);
        setDashboardData(data);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
        setLoading(false);
      });

    // Check if model exists before fetching feature importance
    fetch("http://127.0.0.1:5000/check_model")
      .then((response) => response.json())
      .then((data) => {
        if (data.model_exists) {
          // Fetch Feature Importance Data only if model exists
          fetch("http://127.0.0.1:5000/feature_importance")
            .then((response) => response.json())
            .then((data) => {
              const formattedData = data.feature_importance.map((item) => ({
                ...item,
                feature: formatFeatureName(item.feature), // Format feature names
                importance: item.importance * 100, // Convert to percentage
              }));
              setFeatureImportance(formattedData);
            })
            .catch((error) => {
              console.error("Error fetching feature importance:", error);
            });
        }
      })
      .catch((error) => {
        console.error("Error checking model existence:", error);
      });

    // Load the most recent prediction from local storage
    const savedPrediction = localStorage.getItem("mostRecentPrediction");
    if (savedPrediction) {
      setPredictions([JSON.parse(savedPrediction)]);
    }
  }, []);

  // Handle File Drop
  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file && file.type === "text/csv") {
      setSelectedFile(file);
    } else {
      alert("Please upload a valid CSV file.");
    }
  };

  // Handle File Selection
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type === "text/csv") {
      setSelectedFile(file);
    } else {
      alert("Please upload a valid CSV file.");
      setSelectedFile(null);
    }
  };

  // Handle Prediction Generation & Redirect
  const handleGenerateNewPrediction = async () => {
    if (!selectedFile) return;

    const confirmGenerate = window.confirm(
      "Are you sure you want to generate predictions for this file?"
    );
    if (!confirmGenerate) return;

    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch("http://127.0.0.1:5000/predict_batch", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      // if (response.ok) {
      //   const RISK_THRESHOLD = 0.5;
      //   const riskyCustomers = data.predictions.filter(
      //     (p) => p.churn_probability > RISK_THRESHOLD
      //   );

        if (response.ok) {
          const riskyCustomers = calculateRiskyCustomers(data.predictions, riskThreshold);

        const newPrediction = {
          fileName: selectedFile.name,
          date: new Date().toLocaleDateString(),
          riskyCustomers: riskyCustomers.length,
          predictions: data.predictions,
        };

        // Save the most recent prediction to local storage
        localStorage.setItem(
          "mostRecentPrediction",
          JSON.stringify(newPrediction)
        );

        setPredictions((prev) => [newPrediction, ...prev]);
        setShowModal(false);
        setSelectedFile(null);
        setIsProcessing(false);

        // Navigate to Prediction Results Page
        navigate(`/${encodeURIComponent(selectedFile.name)}`, {
          state: { predictions: data.predictions },
        });
      } else {
        alert("Error generating predictions.");
      }
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Failed to generate predictions.");
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle Removing a Prediction
  const handleRemovePrediction = (fileName) => {
    const updatedPredictions = predictions.filter(
      (prediction) => prediction.fileName !== fileName
    );
    setPredictions(updatedPredictions);

    // Update local storage
    if (updatedPredictions.length > 0) {
      localStorage.setItem(
        "mostRecentPrediction",
        JSON.stringify(updatedPredictions[0])
      );
    } else {
      localStorage.removeItem("mostRecentPrediction");
    }
  };

  // Define pages with all five graphs
  const pages = dashboardData
    ? [
        {
          title: "Churn and Feature Importance",
          content: (
            <Row>
              <Col md={6}>
                <Card className="shadow p-3">
                  <h5 className="text-center mb-3">Churn Counts per Month</h5>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={dashboardData.churn_counts_per_month}>
                      <XAxis dataKey="month" stroke="#333" />
                      <YAxis stroke="#333" />
                      <Tooltip />
                      <Legend
                        align="right" // Align legend to the right
                        verticalAlign="top" // Position legend at the top
                        layout="horizontal" // Display legend horizontally
                      />
                      <Line
                        type="monotone"
                        dataKey="churn_count"
                        stroke="#ff4d4d"
                        strokeWidth={2}
                        name="Churn Count"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Card>
              </Col>

              <Col md={6}>
                <Card className="shadow p-3">
                  <h5 className="text-center mb-3">Feature Importance</h5>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={featureImportance}
                      layout="vertical"
                      margin={{ left: 20, right: 20 }}
                    >
                      <XAxis type="number" stroke="#333" />
                      <YAxis
                        type="category"
                        dataKey="feature"
                        stroke="#333"
                        interval={0}
                        width={130}
                      />
                      <Tooltip />
                      <Legend
                        align="right" // Align legend to the right
                        verticalAlign="top" // Position legend at the top
                        layout="horizontal" // Display legend horizontally
                      />
                      <Bar
                        dataKey="importance"
                        fill="#0088FE"
                        name="Importance (%)"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              </Col>
            </Row>
          ),
        },
        {
          title: "Age, Activation, and App Usage",
          content: (
            <Row>
              <Col md={4}>
                <Card className="shadow p-3">
                  <h5 className="text-center mb-3">Age Range Distribution</h5>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={dashboardData.age_range_counts}>
                      <XAxis dataKey="range" stroke="#333" />
                      <YAxis stroke="#333" />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="count" fill="#00C49F" name="Count" />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              </Col>

              <Col md={4}>
                <Card className="shadow p-3">
                  <h5 className="text-center mb-3">
                    Activation Counts by Month
                  </h5>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={dashboardData.activation_counts}>
                      <XAxis dataKey="month" stroke="#333" />
                      <YAxis stroke="#333" />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="count" fill="#8884d8" name="Count" />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              </Col>

              <Col md={4}>
                <Card className="shadow p-3">
                  <h5 className="text-center mb-3">App Usage Percentages</h5>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={dashboardData.app_usage_percentages}
                      layout="vertical"
                      margin={{ left: 30, right: 20 }}
                    >
                      <XAxis type="number" stroke="#333" />
                      <YAxis
                        type="category"
                        dataKey="app"
                        stroke="#333"
                        interval={0}
                      />
                      <Tooltip />
                      <Legend />
                      <Bar
                        dataKey="percentage"
                        fill="#FF8042"
                        name="App Usage (%)"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              </Col>
            </Row>
          ),
        },
      ]
    : [];

  return (
    <Container fluid className="mt-4">
      <h1 className="text-center mb-4">Dashboard</h1>

      {loading ? (
        <div className="text-center">
          <Spinner animation="border" role="status" />
        </div>
      ) : (
        <>
          {pages[page].content}
          <Row className="justify-content-center mt-4">
            <Col md={6} className="d-flex justify-content-center">
              <Button
                variant="primary"
                onClick={() => setPage((prev) => Math.max(prev - 1, 0))}
                disabled={page === 0}
                className="me-2"
              >
                <i className="bi bi-chevron-left"></i> Previous
              </Button>
              <Button
                variant="primary"
                onClick={() =>
                  setPage((prev) => Math.min(prev + 1, pages.length - 1))
                }
                disabled={page === pages.length - 1}
              >
                Next <i className="bi bi-chevron-right"></i>
              </Button>
            </Col>
          </Row>
        </>
      )}

      {/* Predictions Section */}
      <Row className="mt-4">
        <Col>
          <Card className="shadow p-4 mb-4">
            <div className="d-flex justify-content-between align-items-center">
              <Card.Title>Most Recent Predictions</Card.Title>
              <Button variant="success" onClick={() => setShowModal(true)}>
                Generate New Prediction
              </Button>
            </div>
            <Table responsive>
              <thead>
                <tr>
                  <th>File Name</th>
                  <th>Date</th>
                  <th># of Risky Customers</th>
                  <th>Total # of Customers</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {isProcessing ? (
                  <tr>
                    <td colSpan={4} className="text-center">
                      <Spinner animation="border" size="sm" /> Generating
                      predictions...
                    </td>
                  </tr>
                ) : predictions.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="text-center text-muted">
                      No predictions available.
                    </td>
                  </tr>
                ) : (
                  predictions.map((prediction, index) => (
                    <tr
                      key={index}
                      className={
                        prediction.riskyCustomers > 10 ? "table-warning" : ""
                      }
                    >
                      <td>
                        <Link
                          to={`/${prediction.fileName}`}
                          state={{ predictions: prediction.predictions }}
                        >
                          {prediction.fileName}
                        </Link>
                      </td>
                      <td>{prediction.date}</td>
                      <td>{prediction.riskyCustomers}</td>
                      <td>{prediction.predictions.length}</td>
                      <td>
                        <Button
                          variant="link"
                          className="text-danger p-0"
                          onClick={() =>
                            handleRemovePrediction(prediction.fileName)
                          }
                        >
                          <FaTimes /> {/* X icon */}
                        </Button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </Table>
          </Card>
        </Col>

        {/* Modal for File Upload */}
        <Modal show={showModal} onHide={() => setShowModal(false)}>
          <Modal.Header closeButton>
            <Modal.Title>Upload CSV File</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            {/* Drag-and-Drop Box */}
            <div
              className={`border p-4 rounded bg-light text-muted ${
                dragging ? "border-primary bg-white shadow-lg" : "border-dashed"
              }`}
              style={{ cursor: "pointer" }}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => document.getElementById("fileInput").click()}
            >
              <input
                type="file"
                id="fileInput"
                accept=".csv"
                onChange={handleFileSelect}
                style={{ display: "none" }}
              />
              {selectedFile ? (
                <div className="d-flex justify-content-between align-items-center">
                  <p className="fw-bold text-success mb-0">
                    {selectedFile.name}
                  </p>
                  <Button
                    variant="link"
                    className="text-danger p-0"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedFile(null);
                    }}
                  >
                    Remove
                  </Button>
                </div>
              ) : (
                <p className="mt-2 mb-0">
                  Drag & drop a CSV file here or click to select
                </p>
              )}
            </div>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setShowModal(false)}>
              Close
            </Button>
            <Button
              variant="primary"
              onClick={handleGenerateNewPrediction}
              disabled={!selectedFile || isProcessing}
            >
              {isProcessing ? (
                <Spinner as="span" animation="border" size="sm" />
              ) : (
                "Generate Prediction"
              )}
            </Button>
          </Modal.Footer>
        </Modal>
      </Row>
    </Container>
  );
}
