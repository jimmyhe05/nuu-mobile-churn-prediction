import React, { useState } from "react";
import {
  Container,
  Row,
  Col,
  Card,
  Button,
  Spinner,
  Alert,
} from "react-bootstrap";
import { useNavigate } from "react-router-dom";
import Papa from "papaparse";

function CSVTrainer() {
  const [csvFile, setCsvFile] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  const [modelAccuracy, setModelAccuracy] = useState(null); // Store accuracy result
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState(null); // For error handling
  const navigate = useNavigate();

  // Handle file drop
  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];

    if (file && file.type === "text/csv") {
      setCsvFile(file);
      parseCSV(file);
    } else {
      alert("Please upload a valid CSV file.");
    }
  };

  // Handle file selection via click
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type === "text/csv") {
      setCsvFile(file);
      parseCSV(file);
    } else {
      alert("Please upload a valid CSV file.");
    }
  };

  // Parse CSV file (Dummy logic, replace with actual processing)
  const parseCSV = (file) => {
    Papa.parse(file, {
      complete: () => {
        console.log("CSV file parsed successfully.");
      },
      header: true,
      skipEmptyLines: true,
    });
  };

  // Train the model using the API
  const trainModel = async () => {
    if (!csvFile) {
      alert("Please upload a CSV file first.");
      return;
    }

    setIsTraining(true);
    setTrainingResult(null);
    setModelAccuracy(null); // Reset accuracy before new training
    setError(null); // Clear any previous errors

    const formData = new FormData();
    formData.append("file", csvFile);

    try {
      const response = await fetch("http://127.0.0.1:5000/train_model", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setTrainingResult(data.message); // Display success message
        if (data.metrics) {
          setModelAccuracy({
            precision: data.metrics.precision,
            recall: data.metrics.recall,
            accuracy: data.metrics.accuracy,
            f2_score: data.metrics.F2_score, // Make sure this matches the API response
          });
        }
      } else {
        const errorData = await response.json();
        setError(errorData.error || "Training failed. Please try again.");
      }
    } catch (err) {
      setError("An error occurred while training the model.");
      console.error("Training error:", err);
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <Container className="my-4">
      <Row className="justify-content-center">
        <Col md={8}>
          <Card className="shadow-lg">
            <Card.Body className="text-center">
              <h3 className="mb-3">Upload CSV & Train AI Model</h3>

              {/* Drag-and-Drop Box */}
              <div
                className={`border p-4 rounded bg-light text-muted ${dragging
                    ? "border-primary bg-white shadow-lg"
                    : "border-dashed"
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
                {csvFile ? (
                  <p className="fw-bold text-success">{csvFile.name}</p>
                ) : (
                  <p className="mt-2">
                    Drag & drop a CSV file here or click to select
                  </p>
                )}
              </div>

              {/* Train Button */}
              <Button
                style={{
                  backgroundColor: "#C3D831",
                  borderColor: "#A8C520",
                  color: "white",
                }}
                className="mt-3 px-4"
                onClick={trainModel}
                disabled={isTraining}
              >
                {isTraining ? (
                  <Spinner as="span" animation="border" size="sm" />
                ) : (
                  "Train Model"
                )}
              </Button>

              {/* Error Message */}
              {error && (
                <Alert variant="danger" className="mt-3">
                  {error}
                </Alert>
              )}

              {/* Training Results & Model Accuracy */}
              {trainingResult && (
                <div className="mt-3">
                  <p className="fw-bold text-success">{trainingResult}</p>
                  {/* Display model analytics only if available */}
                  {modelAccuracy && (
                    <div className="text-start mt-3">
                      <h5>📊 Model Analytics:</h5>
                      <ul className="list-group">
                        <li className="list-group-item">
                          <strong>Precision:</strong> {modelAccuracy.precision}
                        </li>
                        <li className="list-group-item">
                          <strong>Recall:</strong> {modelAccuracy.recall}
                        </li>
                        <li className="list-group-item">
                          <strong>Accuracy:</strong> {modelAccuracy.accuracy}
                        </li>
                        <li className="list-group-item">
                          <strong>F2 Score:</strong> {modelAccuracy.f2_score}
                        </li>
                      </ul>
                    </div>
                  )}
                  <Button variant="secondary" onClick={() => navigate("/")}>
                    Go to Dashboard
                  </Button>
                </div>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}

export default CSVTrainer;
