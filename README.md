# diabetes-ucidata

## 📝 Table of Contents
[Data](#data)
[Goal](#goal)

## 📊 Data

[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008?fbclid=IwAR1K8yIAY03mM8Ipm6UQMjX5hW4hr3xbvKneoqDNR-93l2WPCqrXBjl59iM)

The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes. Information was extracted from the database for encounters that satisfied the following criteria.

(1) It is an inpatient encounter (a hospital admission).
(2) It is a diabetic encounter, that is, one during which any kind of diabetes was entered to the system as a diagnosis.
(3) The length of stay was at least 1 day and at most 14 days.
(4) Laboratory tests were performed during the encounter.
(5) Medications were administered during the encounter.

## 📈 Goal
We are trying to predict whether a patient will be readmitted to the hospital (within or after 30 days of discharge). This is a classification problem. We will use the following models:
- Logistic Regression (Just for the readmission)
- Decision Tree
- Random Forest
- XGBoost
- Neural Network

## 📁 Files

