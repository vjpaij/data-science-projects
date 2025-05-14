## 1\. Project Overview

**Objective:** Predict customer churn (whether a customer will leave a service) using machine learning/deep learning.  
**Business Impact:** Helps companies reduce customer attrition by identifying at-risk customers early.

**Key Steps:**
1.  **Data Loading & Exploration**
2.  **Data Cleaning & Preprocessing**
3.  **Exploratory Data Analysis (EDA) & Visualization**
4.  **Feature Engineering & Selection**
5.  **Model Building (Traditional ML & Deep Learning with PyTorch)**
6.  **Hyperparameter Tuning & Model Evaluation**
7.  **Deployment (Flask/Streamlit + Docker + Kubernetes)**
8.  **Monitoring (Prometheus + Grafana Dashboard)**
    

* * *

## 2\. Tools & Technologies Used

| Category | Tools/Libraries |
| --- | --- |
| Programming | Python 3.9+ |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| ML/DL | Scikit-learn, PyTorch (with CUDA support) |
| Deployment | Flask, Streamlit, Docker, Kubernetes |
| Monitoring | Prometheus, Grafana |
| Version Control | Git, GitHub |
| Cloud | AWS/GCP (Optional) |

* * *

## 3\. Dataset Dictionary

We’ll use the **Telco Customer Churn** dataset from Kaggle ([Download Here](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)).

| Column | Description | Data Type |
| --- | --- | --- |
| customerID | Unique customer identifier | Object |
| gender | Male/Female | Categorical |
| SeniorCitizen | Whether customer is a senior (1/0) | Binary |
| Partner | Has a partner (Yes/No) | Categorical |
| Dependents | Has dependents (Yes/No) | Categorical |
| tenure | Months with company | Numeric |
| PhoneService | Has phone service (Yes/No) | Categorical |
| MultipleLines | Multiple phone lines | Categorical |
| InternetService | Type of internet service | Categorical |
| OnlineSecurity | Has online security | Categorical |
| OnlineBackup | Has online backup | Categorical |
| DeviceProtection | Has device protection | Categorical |
| TechSupport | Has tech support | Categorical |
| StreamingTV | Has streaming TV | Categorical |
| StreamingMovies | Has streaming movies | Categorical |
| Contract | Contract type (Month-to-month, 1/2 year) | Categorical |
| PaperlessBilling | Paperless billing (Yes/No) | Categorical |
| PaymentMethod | Payment method | Categorical |
| MonthlyCharges | Monthly charges | Numeric |
| TotalCharges | Total charges (may have missing values) | Numeric |
| Churn | Target variable (Yes/No) | Binary |

* * *

## 4\. How to Run the Code
1. Clone this repository
2. Install requirements: python -m venv venv -> source venv/bin/activate -> `pip install -r requirements.txt`
3. Open and run the Jupyter notebook
4. Run flask app: python churn_app.py

* * *

## 5\. Additional Learning Resources (Udemy Courses)

| Course | Link |
| --- | --- |
| Data Science & Machine Learning | https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/ |
| Deep Learning with PyTorch | https://www.udemy.com/course/pytorch-for-deep-learning/ |
| Docker & Kubernetes | https://www.udemy.com/course/docker-and-kubernetes-the-complete-guide/ |
