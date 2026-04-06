# 🔥 BurnoutIQ — Employee Burnout Risk Analyzer

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://burnoutiq.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> An AI-powered web application that predicts employee burnout risk using machine learning — built with Random Forest Regression, deployed on Streamlit Cloud.

---

## 🌐 Live Demo

👉 **[Try the app here → burnoutiq.streamlit.app](https://burnoutiq.streamlit.app)**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Dataset](#-dataset)
- [Machine Learning Models](#-machine-learning-models)
- [App Walkthrough](#-app-walkthrough)
- [Project Structure](#-project-structure)
- [Installation & Local Setup](#-installation--local-setup)
- [Deployment](#-deployment)
- [Results](#-results)
- [Tech Stack](#-tech-stack)

---

## 📖 Overview

**BurnoutIQ** is a machine learning-powered HR analytics tool that predicts an employee's **burnout risk score** (0–10) based on workplace productivity metrics, workload patterns, and well-being indicators.

The app helps HR teams and managers identify at-risk employees early — before burnout escalates — so they can take timely, data-driven action.

---

## 🎯 Problem Statement

Modern workplaces face a growing burnout crisis. Traditional HR approaches rely on annual surveys or manager intuition, which are slow and often miss early warning signs.

This project addresses:

- How do AI tool adoption, workload, collaboration patterns, and well-being influence burnout?
- Can we build an ML model that predicts burnout risk from employee-level data?
- Can this be deployed as a usable, real-time tool for HR teams?

---

## ✨ Features

- 🔮 **Instant Burnout Score Prediction** — enter 13 employee metrics and get a risk score (0–10) in seconds
- 🟢🟡🔴 **Risk Level Classification** — automatically categorized as Low, Moderate, or High risk
- 💡 **Actionable Recommendations** — each result includes a tailored HR action recommendation
- 📊 **Visual Score Bar** — animated progress bar showing risk intensity
- ♻️ **Reset to Defaults** — one-click reset to run a fresh prediction
- 🌙 **Dark Professional UI** — clean, modern design built for HR dashboards
- 📱 **Responsive Layout** — works on desktop and tablet

---

## 📂 Dataset

Two datasets were used in this project:

| File | Description | Rows | Target |
|------|-------------|------|--------|
| `ai_productivity_features.csv` | Employee workplace metrics | 4,500 | `burnout_risk_score` (0–10, continuous) |
| `ai_productivity_targets.csv` | Employee productivity data | 4,500 | `burnout_risk_level` (Low/Medium/High) |

### Input Features (Regression Model)

| Feature | Type | Description |
|---------|------|-------------|
| `job_role` | Categorical | Employee's role (Data Scientist, Developer, HR, etc.) |
| `experience_years` | Integer | Years of professional experience |
| `deadline_pressure_level` | Ordinal | Low / Medium / High |
| `work_life_balance_score` | Integer (1–10) | Self-reported work-life balance |
| `ai_tool_usage_hours_per_week` | Float | Hours/week using AI tools |
| `manual_work_hours_per_week` | Float | Hours/week on manual tasks |
| `meeting_hours_per_week` | Float | Hours/week in meetings |
| `collaboration_hours_per_week` | Float | Hours/week collaborating |
| `learning_time_hours_per_week` | Float | Hours/week on learning & development |
| `focus_hours_per_day` | Float | Deep work hours per day |
| `tasks_automated_percent` | Float | % of tasks automated |
| `error_rate_percent` | Float | % of tasks with errors |
| `task_complexity_score` | Integer (1–10) | Complexity of assigned tasks |

---

## 🤖 Machine Learning Models

### Regression — Burnout Risk Score

Predicts a continuous burnout score from 0 to 10.

| Model | MAE | MSE | R² Score |
|-------|-----|-----|----------|
| **Random Forest (Tuned)** | **best** | **best** | **best** ✅ |

**Best Model: Random Forest Regressor** (selected via GridSearchCV with 5-fold cross-validation)

Preprocessing pipeline:
1. Drop `Employee_ID`
2. Ordinal encode `deadline_pressure_level` (Low=0, Medium=1, High=2)
3. One-hot encode `job_role` (drop_first=True)
4. StandardScaler on all numeric features
5. Train/test split: 80/20

### Classification — Burnout Risk Level

Predicts Low / Medium / High burnout category.

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **SVM (RBF, Tuned)** | **best** | **best** | **best** | **best** ✅ |

**Best Model: SVM with RBF kernel** (tuned via GridSearchCV Pipeline)

---

## 🖥️ App Walkthrough

### Step 1 — Fill in Employee Details
The app is divided into 3 input sections:

- **👤 Employee Profile** — Job role, experience, deadline pressure, work-life balance
- **⏱ Weekly Time Allocation** — AI tool usage, manual work, meetings, collaboration, learning, focus hours
- **📊 Performance Indicators** — Tasks automated, error rate, task complexity

### Step 2 — Click "Predict Burnout Risk Score"
The Random Forest model runs instantly and returns:

- A **numeric score** (e.g., `6.84 / 10`)
- A **color-coded risk level** (🟢 Low / 🟡 Moderate / 🔴 High)
- An **animated fill bar** showing intensity
- A **recommended HR action** based on the risk level

### Step 3 — Reset & Predict Again
Click **↺ Reset to Defaults** to clear all fields and run a new prediction from scratch.

---

## 📁 Project Structure

```
AI-workplace-productivity/
│
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
│
├── burnout_rate_scores.ipynb     # Regression model notebook
├── burnout_risk_level.ipynb      # Classification model notebook
│
├── ai_productivity_features.csv  # Regression dataset
├── ai_productivity_targets.csv   # Classification dataset
│
├── burnout_random_forest.joblib  # Saved Random Forest model
├── scaler.joblib                 # Saved StandardScaler (regression)
├── rf_columns.joblib             # Saved feature column order
│
├── best_svm_pipeline.pkl         # Saved SVM pipeline (classification)
├── naive_bayes_model.pkl         # Saved Naive Bayes model
├── scaler.pkl                    # Saved scaler (classification)
├── label_encoder.pkl             # Saved LabelEncoder
├── columns.pkl                   # Saved classification columns
│
└── README.md
```

---

## ⚙️ Installation & Local Setup

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Aishwarya-J05/AI-workplace-productivity.git
cd AI-workplace-productivity

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

### Requirements

```
streamlit
scikit-learn
pandas
numpy
```

---

## 🚀 Deployment

The app is deployed on **Streamlit Community Cloud** (free tier).

To deploy your own version:

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"Create app"** → **"Deploy a public app from GitHub"**
4. Set:
   - Repository: `your-username/AI-workplace-productivity`
   - Branch: `main`
   - Main file path: `app.py`
5. Click **Deploy**

---

## 📊 Results

The deployed app uses a **Random Forest Regressor** trained on 4,500 employee records with 13 input features.

Risk thresholds used in the app:

| Score Range | Risk Level | Action |
|-------------|------------|--------|
| 0.0 – 3.9 | 🟢 Low Risk | Routine quarterly monitoring |
| 4.0 – 6.9 | 🟡 Moderate Risk | Workload review within 30 days |
| 7.0 – 10.0 | 🔴 High Risk | Immediate HR intervention |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.9+ |
| **ML Library** | scikit-learn |
| **Models** | Random Forest, SVM, KNN, Naive Bayes, Logistic Regression |
| **Web App** | Streamlit |
| **Data** | pandas, NumPy |
| **Visualization** | Custom CSS + HTML in Streamlit |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | GitHub |

---

## 👩‍💻 Author

**Aishwarya-J05**
- GitHub: [@Aishwarya-J05](https://github.com/Aishwarya-J05)

---

## 📄 License

This project is licensed under the MIT License.

---

> ⚠️ **Disclaimer:** This tool is intended for internal HR analytics purposes only. It is not a substitute for professional psychological assessment or medical advice.
