# 🧪 Urine Analysis Based Diet Recommendation System

## 📌 Overview

This project is a machine learning based web application that analyzes urine test parameters and classifies a patient into different health conditions. Based on the predicted condition, the system provides personalized diet recommendations.

The goal of this project is to convert medical parameter data into actionable lifestyle advice.

---

## 🚀 Features

- 🧠 Machine Learning based classification (Random Forest)
- 📊 Uses real-world medical dataset (Kidney Disease dataset)
- 🧾 Classifies into 5 health categories:
  - Normal
  - High Blood Sugar
  - Kidney Stress
  - Dehydration
  - Infection Risk
- 🥗 Provides condition-specific diet recommendations
- 🌐 Web-based interface for easy interaction

---

## 🧠 How It Works

### 1. Data Preprocessing
- Missing values (`?`) are replaced with NaN
- All values are converted to numeric format
- Missing values are filled using the most frequent value (mode)

### 2. Label Generation
Custom labels are created based on medical parameters:

- **High Blood Sugar** → High glucose or urine sugar  
- **Kidney Stress** → High albumin, creatinine, or urea  
- **Dehydration** → Low specific gravity  
- **Infection Risk** → Low hemoglobin  
- **Normal** → None of the above conditions  

### 3. Model Training
- Model used: **Random Forest Classifier**
- The model learns patterns from labeled data
- Achieves high accuracy on test data

### 4. Prediction
- User inputs urine parameters through UI
- Model predicts condition
- System returns:
  - Predicted health condition
  - Risk level
  - Diet recommendations

---

## 🥗 Diet Recommendation Logic

Each predicted condition is mapped to a specific diet plan:

- **High Blood Sugar**
  - Reduce rice and sugar
  - Eat more whole grains and fiber

- **Kidney Stress**
  - Reduce salt and protein
  - Avoid processed foods

- **Dehydration**
  - Increase water intake
  - Consume electrolyte-rich fluids

- **Infection Risk**
  - Increase vitamin C intake
  - Boost immunity with healthy foods

- **Normal**
  - Maintain a balanced diet and healthy lifestyle

---

## 🛠️ Tech Stack

- **Backend:** Python (Flask)
- **Machine Learning:** scikit-learn
- **Frontend:** HTML/CSS/JavaScript
- **Deployment:** Vercel (Serverless Functions)
