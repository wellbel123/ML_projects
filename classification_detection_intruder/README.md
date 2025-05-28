# 👤 Catch Me If You Can: Alice Session Detection

> Binary classification project: identifying whether a user is **Alice** based on their internet session data.

---

## 🧠 Problem Statement

This is a **binary classification task** where each sample is a user session (a sequence of visited websites with timestamps).  
The goal is to **predict whether the session belongs to Alice** - a specific target user.

I used a data from the [Kaggle competition](https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2).

---

## 📊 Evaluation Metric

**ROC-AUC** is used because:

- it's threshold-independent
- handles class imbalance well
- measures the model’s ranking ability

---

## 🏗️ Approach

1. **Feature Engineering**:
   - Number of sites in session
   - Number of unique sites
   - Session duration
   - Start hour, weekday, weekend indicator
   - Share of Alice’s top-10 websites
   - Cyclic time features (`sin/cos` for hour)

2. **Text Modeling**:
   - Websites in a session are treated as a string sequence
   - `CountVectorizer`
   - Combined with engineered numerical features

3. **Modeling**:
   - Final model: **Logistic Regression** with regularization `C=1.0`
   - Compared with Random Forest, which underperforms due to sparse input

---

## 🚀 Results

| Model                 | ROC-AUC (Validation) |
|----------------------|----------------------|
| Logistic Regression  | **0.9775**           |
| Random Forest        | ~0.93                |

---