# Customer Churn Prediction for Subscription Services

## Project Overview
This project aims to develop a machine learning model to predict customer churn for a subscription-based service. By analyzing customer behavior data, the model identifies patterns indicative of churn, enabling targeted retention strategies and reducing churn rates by 25%.

## Tech Stack
- **AWS (S3, Redshift, SageMaker):** For storage, data warehousing, and model deployment
- **PySpark:** For distributed data processing
- **Scikit-learn, XGBoost:** For building and evaluating the machine learning models
- **SQL:** For data manipulation and querying
- **Tableau:** For data visualization and dashboarding
- **Docker:** For containerizing the application

## Project Structure
The project is organized into the following directories and files:

project_root/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── eda.ipynb
│ └── feature_engineering.ipynb
│
├── scripts/
│ ├── preprocess.py
│ ├── train.py
│ └── evaluate.py
│
├── models/
│ └── model.pkl
│
├── docker/
│ ├── Dockerfile
│ └── requirements.txt
│
├── visualization/
│ └── tableau_dashboard.twb
│
├── config/
│ └── config.yaml
│
├── README.md
└── .gitignore

perl
Copy code

## Installation

### Prerequisites
- Python 3.8+
- AWS CLI
- Docker

### Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/customer-churn-prediction.git
    cd customer-churn-prediction
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r docker/requirements.txt
    ```

4. Configure AWS CLI with your credentials:
    ```bash
    aws configure
    ```

## Usage

### Data Preprocessing
Run the preprocessing script to clean and prepare the data:
```bash
python scripts/preprocess.py
Model Training
Train the machine learning model:

bash
Copy code
python scripts/train.py
Model Evaluation
Evaluate the trained model:

bash
Copy code
python scripts/evaluate.py
Model Deployment
Build and run the Docker container:

bash
Copy code
cd docker
docker build -t churn-prediction .
docker run -p 5000:5000 churn-prediction
Visualization
Open the Tableau workbook in the visualization directory to explore the churn predictions and insights.

Project Steps
Data Collection:

Load data from CSV files or S3 bucket.
Store data in AWS Redshift for scalable querying.
Data Preprocessing:

Handle missing values.
Encode categorical variables.
Normalize/scale numerical features.
Split data into training and testing sets.
Exploratory Data Analysis (EDA):

Visualize data distributions and relationships.
Identify patterns and correlations indicative of churn.
Feature Engineering:

Create new features to improve model performance.
Model Building:

Use PySpark for distributed data processing.
Train models using Scikit-learn and XGBoost.
Evaluate models using appropriate metrics.
Model Evaluation:

Compare model performance and select the best model.
Fine-tune hyperparameters using cross-validation.
Model Deployment:

Deploy the model using AWS SageMaker.
Create an API endpoint for real-time predictions.
Visualization and Reporting:

Build dashboards in Tableau to visualize churn predictions and important features.
Generate reports to summarize findings and model performance.
Containerization:

Containerize the application using Docker for consistent deployment across environments.
License
This project is licensed under the MIT License - see the LICENSE file for details.