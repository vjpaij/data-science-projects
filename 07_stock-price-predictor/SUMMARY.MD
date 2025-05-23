## 1\. Project Overview

This project builds an end-to-end Stock Price Predictor using historical stock data. It includes complete steps from data collection, preprocessing,
EDA, feature engineering, model training, hyperparameter tuning, evaluation, and deployment using Streamlit.
We also use PyTorch for deep learning models with GPU support.

* * *

## 2\. Tools & Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn, Plotly
- PyTorch (with GPU support)
- yfinance (for stock data)
- Streamlit (for deployment)
- joblib (for model persistence)
- tqdm (for progress bars)
- VS Code / Jupyter Notebook (IDE)

* * *

## 3\. Dataset Dictionary

Using `yfinance`, columns typically include:
- Date: Timestamp
- Open: Opening price of the stock
- High: Highest price
- Low: Lowest price
- Close: Closing price
- Adj Close: Adjusted closing price
- Volume: Number of shares traded

* * *

## 4\. How to Run the Code
1. Clone this repository
2. Install requirements: python -m venv venv -> source venv/bin/activate -> `pip install -r requirements.txt`
3. Open and run the Jupyter notebook
4. Run streamlit app: streamlit run stock_app.py
5. Run Prometheus: ./prometheus --config.file=prometheus.yml. Open http://localhost:9090
6. Build and run the Docker image: docker build -t stock-price-predictor -> docker run -p 8501:8501 stock-price-predictor
7. To check monitoring, we can go with dockor-compose: docker-compose up --build

**Access Services**:
Streamlit App - http://localhost:8501
Prometheus UI -	http://localhost:9090
Grafana UI - http://localhost:3000
* * *

## 5\. Additional Learning Resources (Udemy Courses)

- Python for Data Science and Machine Learning Bootcamp by Jose Portilla
- PyTorch for Deep Learning and Computer Vision by Jose Portilla
- Data Science and Machine Learning A-Z by Kirill Eremenko
