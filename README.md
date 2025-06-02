# Rainfall Prediction – Kaggle Playground Series S5E3 (Top 2%)

This repository contains the solution to the Kaggle competition:
**Playground Series - Season 5, Episode 3: Binary Prediction with a Rainfall Dataset**

🏆 **Result: Top 2% – Rank 64 out of 4381 teams**

---

## 🔍 Overview
This project focuses on predicting rainfall occurrence (binary classification) using a structured weather dataset. The full pipeline includes:

- 📊 Exploratory Data Analysis (EDA)
- 🧠 Feature Engineering (custom transformers)
- ⚖️ Class balancing with **SMOTE**
- 🔁 Log transformations and **scaling**
- 🧪 Training multiple classifiers: **Logistic Regression, Random Forest, CatBoost, XGBoost**
- 🧵 Cross-validation with **StratifiedKFold** and **ROC AUC** scoring
- 🔍 **GridSearchCV** for hyperparameter tuning
- 🧬 Final ensemble averaging of probability predictions

---

## 🧰 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- CatBoost, XGBoost
- Matplotlib, Seaborn
- Imbalanced-learn (SMOTE)

### 📦 requirements.txt
```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
catboost
imbalanced-learn
```

---

## 🧪 Model Evaluation
Models were evaluated using 5-fold cross-validation and ROC AUC metric.
Best-performing single models (AUC > 0.89) were averaged into an ensemble.

---

## 📂 Structure
```
├── notebook.ipynb              # Full code with explanation (EDA + modeling)
├── pipeline.py                 # Optional: modular pipeline components
├── requirements.txt            # Dependencies
├── README.md                   # This file
```

---

## 📊 Highlights
- `avg_temp` engineered feature
- Seasonal feature from `day`
- Log-transformation of skewed features (e.g., dewpoint)
- ROC AUC ~ **0.89+** consistently across models

---

## 🚀 Result
Top 2% on Kaggle leaderboard with final ROC AUC: **~0.89**

📌 Competition link: https://www.kaggle.com/competitions/playground-series-s5e3

---

## 👤 Author
Project by **Maryna Dudik**  
🧑‍💻 GitHub: [MarkLewkowskii](https://github.com/MarkLewkowskii)  
🏅 Kaggle: [marinadudik](https://www.kaggle.com/marinadudik)



