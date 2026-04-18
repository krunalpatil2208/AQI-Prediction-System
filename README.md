# AQI Prediction & Analytics System

This project predicts Air Quality Index (AQI) using historical data (2015–2025) and provides interactive analytics through a web-based dashboard.

---

##  Tech Stack
- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS, JavaScript
- **Visualization:** Chart.js, Plotly
- **Machine Learning:** XGBoost Regressor



##  Features
- AQI Prediction based on **state, area, month, and year**
- AQI category classification (Good, Moderate, Poor, etc.)
- Interactive analytics dashboard:
  - Top polluted areas
  - Top polluted states
  - AQI category distribution
  - Monthly AQI trends
  - Heatmap (state vs month)

---

##  Model
- Algorithm: XGBoost Regressor
- Evaluation Metric: R² Score
- Preprocessing:
  - Date → Year & Month extraction
  - Label Encoding for state & area

---

##  Pre-trained Model
The project uses a pre-trained model (`aqi_model.pkl`) and encoders (`state_encoder.pkl`, `area_encoder.pkl`).

 No training required — just run the application.

---

##  How to Run

### 1. Clone the repository

- git clone https://github.com/krunalpatil2208/AQI-Prediction-System.git
- cd AQI-Prediction-System

### 2. Create virtual environment (recommended)

- python -m venv venv
- venv\Scripts\activate

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Run the Application

python app.py or py app.py

### 5. Open in Browser

http://127.0.0.1:5000


---

##  Dataset
- Historical AQI dataset (2015–2025)
- Stored in `cleaned_aqi.csv`

---

##  Author
- https://github.com/krunalpatil2208