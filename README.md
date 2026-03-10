# Heart-Disease-Prediction-Classifier

A machine learning project that predicts the presence of heart disease using a **Decision Tree Classifier**, achieving **80.3% accuracy** with an optimal tree depth of **4**.

---

## 🔍 Overview

This project applies a **Decision Tree Classifier** to predict whether a patient has heart disease (binary classification: 0 = No Disease, 1 = Disease).

The key finding is that a `max_depth` of **4** provides the best balance between model complexity and accuracy, avoiding overfitting while maintaining strong predictive performance.

---

## 📊 Dataset 
**Rows:** 303 | **Features:** 13 | **Target:** `target` (0 or 1)

| Feature | Description |
|---------|-------------|
| `age` | Age of the patient |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) |
| `thal` | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversable defect) |
| `target` | Diagnosis result (1 = disease, 0 = no disease) |

---

## 📁 Project Structure

```
heart-disease-decision-tree/
│
├── heart.csv                        # Dataset
├── heart_disease_decision_tree.ipynb # Jupyter Notebook (full analysis)
├── README.md                        # Project documentation
├── images/
│   ├── performance_plots.png        # Accuracy vs Depth + Confusion Matrix
│   ├── decision_tree.png            # Tree visualization
│   └── feature_importance.png      # Feature importances bar chart
└── requirements.txt                 # Python dependencies

---

## 📈 Results

### Accuracy vs Max Depth

| Max Depth | Accuracy |
|-----------|----------|
| 1 | 72.13% |
| 2 | 73.77% |
| 3 | 78.69% |
| **4** | **80.33%** ✅ Best |
| 5 | 80.33% |
| 6 | 77.05% |
| 7–13 | ~77.05% |

> **Best max_depth = 4** with accuracy **~80.3%**

### Classification Report (max_depth=4)

```
              precision    recall  f1-score   support

   No Disease     0.74      0.90      0.81        29
      Disease     0.88      0.72      0.79        32

    accuracy                           0.80        61




## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** — data manipulation
- **scikit-learn** — machine learning (DecisionTreeClassifier)
- **matplotlib / seaborn** — data visualization
- **Jupyter Notebook** — interactive development

---
