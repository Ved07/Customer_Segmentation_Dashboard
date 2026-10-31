# Customer Segmentation Dashboard

A Streamlit-based dashboard for exploring customer data and applying K-Means clustering to segment customers based on behavioral attributes like annual income and spending score.

## Project Overview

This project analyzes a customer dataset to:

- inspect distributions and feature relationships
- clean and preprocess raw data
- handle missing values, duplicates, and outliers
- encode categorical features when needed
- choose clustering features
- train a K-Means model
- compare cluster quality using silhouette metrics
- predict segment labels for new customer profiles

The application is built for quick exploration and demonstration, and it ships with a sample dataset for immediate use.

## Tech Stack

- Python 3.10+
- Streamlit
- Pandas
- NumPy
- Plotly
- scikit-learn

## Project Structure

```text
Customer_Segmentation_Dashboard/
├── app.py                  # Main Streamlit dashboard
├── Data/
│   └── Mall_Customers.csv # Sample customer dataset
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python runtime hint for deployment environments
├── README.md               # Project documentation
└── Docs/
    └── Customer Segmentation with K-Means Clustering.pdf
```

## Features

### Data upload and exploration
- upload a custom CSV file
- preview the dataset
- inspect missing values
- view histograms, box plots, and scatter plots

### Preprocessing
- handle missing values with configurable strategies
- remove outliers using IQR thresholds
- remove duplicate rows
- one-hot encode categorical columns

### Feature selection
- review statistics for numeric features
- choose the exact variables used for clustering

### Model training
- scale features with StandardScaler
- analyze elbow and silhouette metrics
- tune the number of clusters
- train K-Means
- visualize cluster distribution and PCA-based separation

### Prediction and reporting
- attach predicted cluster labels to the dataset
- summarize segment size and composition
- export the result as a CSV file
- predict cluster membership for new customer inputs

## Default Dataset

The project includes a bundled dataset in `Data/Mall_Customers.csv`.

This sample file contains customer attributes such as:

- CustomerID
- Genre
- Age
- Annual Income (k$)
- Spending Score (1-100)

When the app starts without an uploaded CSV, it automatically loads this dataset so the dashboard can be demonstrated immediately.

## Getting Started

1. Open a terminal in the project directory.
2. Create and activate a virtual environment (optional but recommended).
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```

5. Open the local URL displayed in the terminal (typically http://localhost:8501).

## Run in Production-Style Environments

The project includes a `runtime.txt` file for deployment compatibility:

```text
python-3.10
```

## Notes

- This dashboard is designed for demonstration and educational use.
- K-Means clustering is sensitive to scaling and feature choice, so preprocessing and feature selection are critical to obtaining meaningful segments.
- The bundled dataset is a practical example for retail customer segmentation and can be replaced with any CSV containing the appropriate numeric fields.

## License

This project is intended for learning and demonstration purposes.
