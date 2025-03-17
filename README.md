# Nuu Mobile Churn Prediction Project

## Overview
The Nuu Mobile Churn Prediction project is a data-driven web application designed to help Nuu Mobile identify customers at risk of churning. By leveraging machine learning and data visualization, the application provides actionable insights to improve customer retention strategies.

## What It Does

### Churn Prediction
- Predicts the likelihood of customers churning based on their usage patterns, device information, and account details
- Allows users to upload a CSV file containing customer data and generates churn predictions in real-time

### Dynamic Risk Threshold
- Users can adjust the risk threshold (0% to 100%) to identify high-risk customers dynamically
- The number of risky customers updates automatically based on the selected threshold

### Interactive Dashboard
- Visualizes key metrics such as churn counts, feature importance, and customer demographics using interactive charts and graphs
- Provides a user-friendly interface for exploring predictions and insights

### Real-Time Insights
- Displays real-time updates on the number of risky customers and their churn probabilities
- Highlights high-risk customers in a table for easy identification

## Technical Implementation

### Frontend Development
- Built a responsive and interactive user interface using React and React Bootstrap
- Integrated Recharts and Chart.js for data visualization
- Implemented dynamic filtering and sorting of customer data based on churn probability

### Backend Development
- Developed a Flask-based backend to handle data processing and machine learning predictions
- Trained a machine learning model using Scikit-learn to predict customer churn
- Created RESTful APIs for data upload, prediction generation, and dashboard data retrieval

### Machine Learning
- Preprocessed customer data (e.g., handling missing values, feature engineering)
- Trained and evaluated a churn prediction model using historical customer data
- Calculated feature importance to identify key factors influencing churn

### Deployment
- Containerized the application using Docker for easy deployment

### User Experience
- Designed an intuitive drag-and-drop interface for uploading CSV files
- Added a dynamic risk threshold slider and input field for real-time updates
- Ensured the application is fully responsive and works seamlessly across devices

## Key Features
- **Batch Prediction**: Upload a CSV file to generate churn predictions for multiple customers
- **Dynamic Threshold**: Adjust the risk threshold to customize the definition of "risky" customers
- **Data Visualization**: Interactive charts and graphs for exploring churn trends and feature importance
- **Real-Time Updates**: Real-time updates on risky customers and their churn probabilities

## Technologies Used

### Frontend
- React
- React Bootstrap
- Recharts
- Chart.js

### Backend
- Flask
- Scikit-learn
- Pandas
- NumPy

### Database
- PostgreSQL (planned for future development to store past prediction files)

## Getting Started

### Prerequisites
- Node.js (v14 or higher)
- Python (v3.8 or higher)
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd nuu-mobile-churn-prediction
```

2. Install frontend dependencies:
```bash
cd frontend
npm install
```

3. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

### Running the Application

1. Start the frontend development server:
```bash
cd frontend
npm start
```

2. Start the backend server:
```bash
cd backend
python app.py
```

The application will be available at `http://localhost:3000`

## Future Development
- Integration with PostgreSQL for storing historical prediction data
- Enhanced visualization features
- Advanced machine learning model improvements
- API documentation and integration guides

## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details. 