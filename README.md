# Student Performance & At-Risk Prediction

An advisory machine learning decision-support system designed for educational institutions to identify students who may benefit from early academic intervention before major assessments.

<p align="center">
  <img src="reports/dashboard_demo.gif" alt="Student Performance & At-Risk Prediction Dashboard Demo" width="900" />
</p>

---

## 📋 Project Metadata

| Field | Detail |
| :--- | :--- |
| **Domain** | School Management |
| **Assigned Project** | Student Performance & At-Risk Prediction |
| **Prediction Unit** | Individual Student |
| **Prediction Horizon** | 2–4 weeks prior to major/final assessment |
| **Outcome Definition** | Expected score $< 40\%$ on upcoming assessment (`at_risk = 1`) |
| **Model Type** | Supervised Classification (Random Forest with Asymmetric Threshold Tuning) |
| **Inference Interface** | Interactive Streamlit Web Application (`app.py`) |

> [!IMPORTANT]
> **Advisory Use Policy**: This tool is strictly a **decision-support advisory system** intended to help educators offer timely tutoring and counseling. Model outputs are probabilistic indicators and must **never** be used for automated grading, disciplinary actions, tracking, or punitive decisions without human review.

---

## 1. Business Problem & Objective

Early identification of struggling students is critical for preventing course failure and school dropouts. Traditional assessment processes often discover learning gaps only after final exams are graded, when it is too late to intervene.

This project delivers a **reproducible, privacy-preserving machine learning solution** that analyzes historical and behavioral signals to estimate student risk levels well before the final assessment.

### Key Objectives
1. **Maximize Recall on At-Risk Students**: Reduce False Negatives so struggling students are not missed.
2. **Strict Leakage Prevention**: Rely exclusively on behavioral and academic indicators collected prior to the assessment.
3. **Ethical & Fair Design**: Exclude protected demographics (gender, race, socioeconomic status, disability) to prevent algorithmic bias.
4. **Actionable Decision Support**: Expose predicted probabilities alongside concrete intervention recommendations via an intuitive dashboard.

---

## 2. Dataset Description & Generation

To guarantee privacy compliance and complete experimental reproducibility, the model is trained on a controlled **synthetic educational dataset** (`data/processed/student_performance_synthetic.csv`) containing **5,000 student records** generated with a fixed seed (`seed=42`).

### Data Generating Process
Student attributes are generated using a latent engagement variable $z \sim \mathcal{N}(0, 1)$ combined with realistic bounded distributions:

- **Pre-Outcome Input Features**:
  1. `attendance_percentage`: Lecture and tutorial attendance percentage (range: 0–100%).
  2. `assignment_completion`: Coursework homework and practical assignments submitted (range: 0–100%).
  3. `previous_score`: Score achieved on prior baseline assessments (range: 0–100%).
  4. `study_hours`: Average self-reported independent daily study hours (range: 0.0–8.0 hrs).
  5. `participation_score`: Classroom and LMS interaction score (range: 0–100%).
  6. `late_submissions`: Count of assignments submitted after the deadline (range: 0–8).

- **Target Variable (`at_risk`)**:
  - The expected final assessment score is modeled from weighted academic indicators plus Gaussian noise:
    $$\text{Expected Score} = 0.35 \times \text{Attendance} + 0.30 \times \text{Previous Score} + 0.20 \times \text{Assignments} + 0.10 \times \text{Participation} + 0.05 \times \text{Study Hours} - 1.5 \times \text{Late Submissions} + \epsilon$$
  - A student is categorized as **At Risk** (`at_risk = 1`) if their expected score falls below the passing threshold of **40%**, and **Not At Risk** (`at_risk = 0`) otherwise.
  - **Class Distribution**:
    - **Not At Risk (`0`)**: 4,245 students (84.9%)
    - **At Risk (`1`)**: 755 students (15.1%)
    - Reflects real-world academic failure rates with inherent class imbalance.

---

## 3. Machine Learning Pipeline & Model Evaluation

### Data Splitting Strategy
- Split: **80% Training ($N=4,000$)** and **20% Test ($N=1,000$)** with **stratification** on `at_risk`.
- A 20% validation split from the training set was used exclusively for classification threshold calibration, ensuring zero data leakage into the final test evaluation.

### Models Evaluated & Benchmark Comparison
Three candidate architectures were evaluated on the held-out test set ($N_{\text{test}} = 1,000$ with 151 at-risk students):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Standardized Baseline)** | 91.80% | 76.34% | 66.23% | 0.7092 | 0.9234 |
| **Gradient Boosting Classifier** | 91.00% | 72.93% | 64.24% | 0.6831 | 0.9281 |
| **Random Forest (Balanced, Default Threshold 0.50)** | 89.50% | 62.92% | 74.17% | 0.6809 | 0.9312 |
| **Random Forest (Tuned Decision Threshold = 0.35)** 🏆 | **82.90%** | **46.58%** | **90.07%** | **0.6140** | **0.9312** |

---

### Decision Threshold & Cost Asymmetry Justification

In academic early-warning systems, the costs of misclassification are highly asymmetric:
- **False Negative (FN)**: Failing to identify an at-risk student. The student receives no support and may fail the course or drop out (**Critical Cost**).
- **False Positive (FP)**: Flagging a passing student for an advisory check-in. The advisor conducts a brief check-in or offers optional tutoring (**Low Cost**).

By tuning the decision threshold on validation data from `0.50` down to `0.35`, the model achieves **90.07% Recall**, capturing **136 out of 151** struggling students and slashing missed students from 51 down to only 15.

#### Confusion Matrix (Held-out Test Set, $N=1,000$ at Threshold = 0.35)

```
                     Predicted: 0          Predicted: 1
Actual: Not At Risk   TN = 693 (81.6%)     FP = 156 (18.4%)
Actual: At Risk       FN =  15  (9.9%)     TP = 136 (90.1%)
```

---

### Feature Importance (Learned by Random Forest)

The relative importance of signals learned by the production Random Forest model:

| Rank | Feature | Importance | Interpretation |
| :---: | :--- | :---: | :--- |
| 1 | `attendance_percentage` | **27.49%** | Primary engagement metric; chronic absenteeism strongly predicts failure. |
| 2 | `previous_score` | **24.06%** | Prior academic baseline provides foundation knowledge benchmark. |
| 3 | `assignment_completion` | **21.75%** | Consistent coursework completion reflects ongoing study commitment. |
| 4 | `participation_score` | **12.96%** | Classroom and LMS engagement indicator. |
| 5 | `study_hours` | **10.00%** | Self-directed independent learning hours per day. |
| 6 | `late_submissions` | **3.74%** | Behavioral indicator of time management and deadline distress. |

---

## 4. Project Structure

```
risk-pred/
├── .gitignore                      # Git exclusion rules (tracks models and processed data)
├── app.py                          # Interactive Streamlit advisory dashboard
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Pinned dependencies
├── data/
│   ├── Student prediction info.docx# Assignment brief & evaluation rubric
│   ├── processed/
│   │   ├── .gitkeep
│   │   └── student_performance_synthetic.csv # Reproducible dataset (5,000 records)
│   └── raw/
│       └── .gitkeep                # Raw data placeholder
├── models/
│   ├── .gitkeep
│   └── student_risk_model.joblib   # Trained model artifact bundle (threshold 0.35)
├── notebooks/
│   ├── model_training.ipynb        # Data generation, EDA, model training & tuning
│   └── load_model.ipynb            # Model loading verification & inference check
├── reports/
│   ├── .gitkeep                    # Generated evaluation figures
│   └── dashboard_demo.gif          # Demonstration GIF of the Streamlit dashboard
├── src/
│   └── .gitkeep                    # Modular source package
└── tests/
    └── test_model.py               # Automated unit test suite (bundle, threshold, inference)
```

---

## 5. Installation & Environment Setup

### Prerequisites
- Python **3.10** or **3.11** installed.

### Setup Instructions

1. **Clone the repository and enter the directory**:
   ```bash
   git clone https://github.com/Chandra142/student-performance-analytics.git
   cd student-performance-analytics/risk-pred
   ```

2. **Create and activate a virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 6. Running Automated Tests

Run the included automated test suite to verify model artifact integrity, threshold configuration, reproducible data generation, and sample inference:

```bash
python -m unittest discover tests
```

Expected output:
```text
.....
----------------------------------------------------------------------
Ran 5 tests in ~3.2s

OK
```

---

## 7. Launching the Interactive Dashboard

Launch the local Streamlit application:

```bash
streamlit run app.py
```

The app will launch in your default browser at `http://localhost:8501`.

### Interface Capabilities
- **Interactive Student Profiler**: Adjust sliders for attendance, assignment completion, previous scores, daily study hours, participation, and late submissions.
- **Real-Time Risk Probability**: Displays exact estimated risk percentage and progress indicator.
- **Calibrated Advisory Flag**: Alerts when probability meets or exceeds the tuned decision threshold (`0.35`).
- **Targeted Action Plan**: Recommends personalized academic guidance (e.g., peer tutoring, remedial check-ins, study habit consultations).
- **Mandatory Human-in-the-Loop Notice**: Prominently displays advisory disclaimer for compliance.

---

## 8. Responsible AI, Ethical Considerations & Failure Modes

### 1. Human-in-the-Loop Governance
The model outputs probabilistic estimates designed to **assist educators**, never to replace human judgment. All flagged cases require an advisor to review external context (e.g., student health, personal hardship) before reaching out.

### 2. Algorithmic Fairness & Absence of Protected Attributes
The feature pipeline deliberately excludes demographic and socioeconomic variables:
- No gender, race, religion, ethnicity, or nationality.
- No family income, zip code, or parent education level.
- No disability or medical status.
Predictions are grounded solely in observable academic engagement signals.

### 3. Known Limitations & Failure Modes
- **Synthetic Data Calibration**: Trained on synthetic educational data. Deploying in a live institution requires recalibrating thresholds against actual cohort historical failure distributions.
- **Unmeasured Outside Factors**: Sudden illness, family bereavement, or financial distress can lead to sudden drops not anticipated by historical trends.
- **Self-Reported Metric Noise**: Study hours rely on self-reported estimates which may contain reporting bias.

---

## 9. Submission & Acceptance Verification Checklist

- [x] **Reproducible Data Pipeline**: Dataset generated with fixed seed (`seed=42`).
- [x] **Zero Target Leakage**: Only pre-assessment indicators used; outcome defined strictly in the future.
- [x] **Baseline & Candidate Models**: Evaluated Logistic Regression, Gradient Boosting, and Random Forest.
- [x] **Cost-Sensitive Metrics**: Threshold tuned to `0.35` achieving **90.07% Recall**.
- [x] **Working Inference App**: Streamlit dashboard loading serialized `.joblib` bundle.
- [x] **Automated Test Suite**: Unit tests passing under `tests/test_model.py`.
- [x] **Responsible AI Compliance**: Human review policy and ethical fairness considerations documented.
