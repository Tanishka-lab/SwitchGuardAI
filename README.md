# 🚦 SwitchGuardAI

> **Temperature-Based Predictive Maintenance for Railway Switch Machines using Machine Learning**

SwitchGuardAI is an AI-powered predictive maintenance system that predicts faults in railway switch machines using temperature, electrical, operational, and maintenance data. The system helps maintenance engineers identify potential failures before they occur, enabling proactive maintenance and reducing downtime.

---

# 📌 Features

- 🚆 Railway switch machine predictive maintenance
- 🌡 Temperature-based fault prediction
- 📊 Synthetic railway asset dataset generation
- 🔎 Exploratory Data Analysis (EDA)
- 🧹 Data validation & preprocessing
- ⚙️ Advanced feature engineering
- 🌲 Random Forest machine learning model
- 📈 Model evaluation & performance metrics
- 🧠 Explainable AI using SHAP
- 📄 Automatic PDF report generation
- 🖥 Interactive Streamlit dashboard

---

# 🏗 Project Structure

```text
SwitchGuardAI/

├── analysis/
├── configs/
├── dashboard/
├── data/
│   ├── raw/
│   └── processed/
├── feature_engineering/
├── models/
├── reporting/
├── results/
├── saved_models/
├── simulation/
├── preprocess_data.py
├── feature_pipeline.py
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

# 🔄 Project Workflow

```text
Dataset Generation
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Validation
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Train/Test Split
        │
        ▼
Random Forest Training
        │
        ▼
Model Evaluation
        │
        ▼
Explainable AI (SHAP)
        │
        ▼
Interactive Dashboard
```

---

# 🤖 Machine Learning Model

### Algorithm

- Random Forest Classifier

### Prediction Target

- Fault_Label

### Fault Classes

- 🟢 Normal
- 🟡 Maintenance_Overdue
- 🟠 Mechanical_Resistance
- 🟠 Overheat
- 🔴 Critical

---

# 📊 Dashboard Modules

The Streamlit dashboard provides:

- 🏠 Home
- 🎯 Prediction
- 📊 Model Performance
- 🔍 Explainability
- 📄 Report Generation
- ℹ About

---

# 📄 Report Generation

Automatically generates a PDF report containing:

- Asset ID
- Predicted Fault
- Estimated Health Index
- Risk Level
- Maintenance Recommendation
- Next Maintenance Date
- Sensor Readings

---

# 🧠 Explainable AI

SwitchGuardAI uses **SHAP (SHapley Additive Explanations)** to explain model predictions.

Generated explainability outputs include:

- Feature Importance
- SHAP Summary Plot
- SHAP Waterfall Plot
- SHAP Values

---

# 📂 Dataset

The project uses a realistic synthetic dataset containing:

- Ambient Temperature
- Motor Temperature
- Gearbox Temperature
- Lock Temperature
- Control Cabinet Temperature
- Motor Current
- Operation Duration
- Operation Count
- Maintenance History
- Environmental Conditions
- Operational Metadata

---

# 🚀 Quick Start

## 1. Clone the repository

```bash
git clone https://github.com/Tanishka-lab/SwitchGuardAI.git

cd SwitchGuardAI
```

---

## 2. Create a virtual environment

```bash
python -m venv venv
```

---

## 3. Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

# 🔄 Rebuild the Complete Pipeline (Optional)

If you wish to regenerate the dataset and retrain the machine learning model from scratch, run:

```bash
python run_pipeline.py
```

The pipeline performs:

- Dataset Generation
- Exploratory Data Analysis
- Data Validation
- Data Cleaning
- Feature Engineering
- Train/Test Split
- Random Forest Training
- Model Evaluation
- Explainable AI

> **Note:** Running the pipeline regenerates datasets, retrains the model, and updates evaluation results. This step is **optional** because the repository already includes a trained model and processed data required to use the dashboard.

---

# 🛠 Technology Stack

### Programming

- Python

### Machine Learning

- Scikit-learn

### Data Processing

- Pandas
- NumPy

### Explainable AI

- SHAP

### Visualization

- Matplotlib

### Dashboard

- Streamlit

### Report Generation

- ReportLab

### Model Serialization

- Joblib

---

# 📁 Generated Outputs

### Data

- Raw Dataset
- Clean Dataset
- Feature Engineered Dataset
- Train Dataset
- Test Dataset

### Results

- Classification Report
- Confusion Matrix
- ROC Curve
- Feature Importance
- SHAP Summary Plot
- SHAP Waterfall Plot

### Models

- Trained Random Forest Model
- Feature Name Mapping

---

# 🎯 Future Improvements

- Real-time IoT sensor integration
- Live railway asset monitoring
- Cloud deployment
- REST API
- Deep Learning models
- Digital Twin integration

---

# 👩‍💻 Developer

**Tanishka Trehan**

B.Tech Information Technology

---

# 📜 License

This project is developed for educational and research purposes.You are free to use, modify, and distribute this software under the terms of the MIT License.
