# diabetes-ucidata

## 📝 Table of Contents
1. [Data](#📊-data)
2. [Goal](#📈-goal)
3. [Data Collection and Preprocessing the data](#data-collection-and-preprocessing-the-data)

## 📊 Data

[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008?fbclid=IwAR1K8yIAY03mM8Ipm6UQMjX5hW4hr3xbvKneoqDNR-93l2WPCqrXBjl59iM)

The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes. Information was extracted from the database for encounters that satisfied the following criteria.

- It is an inpatient encounter (a hospital admission).
- It is a diabetic encounter, that is, one during which any kind of diabetes was entered to the system as a diagnosis.
- The length of stay was at least 1 day and at most 14 days.
- Laboratory tests were performed during the encounter.
- Medications were administered during the encounter.

## 📈 Goal
We are trying to predict whether a patient will be readmitted to the hospital (within or after 30 days of discharge). This is a classification problem. We will use the following models:
- Logistic Regression (Just for the readmission)
- Decision Tree
- Random Forest
- XGBoost
- Neural Network

## Data Collection and Preprocessing the data
Data Collection

The data collection process involves gathering data from various sources and storing it in a suitable format for further analysis. In this project, the data was collected from the UCI Machine Learning Repository.

The following steps were followed for data collection:

1. Identify the data source: The UCI Machine Learning Repository was chosen as the data source for this project.

2. Access the data: The dataset was downloaded from the UCI Machine Learning Repository website using the provided link.

3. Load the data: The downloaded dataset was loaded into the project using appropriate libraries and functions.

4. Explore the data: The loaded data was explored to understand its structure, features, and any missing values.

5. Clean the data: Data cleaning techniques were applied to handle missing values, outliers, and any inconsistencies in the data.

6. Save the data: The cleaned data was saved in a suitable format (e.g., CSV) for further analysis.


#### FILEPATH: /src/data/data_preprocessing.py

Data Preprocessing

Data preprocessing is an essential step in preparing the data for analysis and modeling. It involves transforming the raw data into a format that is suitable for machine learning algorithms. In this project, the following data preprocessing steps were performed:

1. Handling missing values: Missing values in the dataset were identified and handled using techniques such as imputation or deletion.

2. In the data_preprocessing.py file, we included functions to divide the dataset into two parts based on the presence of the weight column.

The first part of the dataset does not include the weight column. This means that any rows in the original dataset that had a weight value were removed in this subset. This can be useful if you want to analyze or model the data without considering the weight information.

The second part of the dataset includes the weight column, but only includes the rows where the weight values are non-empty. This subset is useful if you specifically want to analyze or model the data that has weight information available.

By dividing the dataset in this way, we can handle the weight column separately or exclude it altogether, depending on the specific requirements of our analysis or modeling tasks.


#### FILEPATH: /src/data/data_scaling.py

Data Scaling

Data scaling is a technique used to standardize the range of features in the dataset. It ensures that all features have a similar scale, which can improve the performance of machine learning algorithms. In this project, the following data scaling techniques were applied:

1. Min-max scaling: This technique scales the features to a specific range, typically between 0 and 1. It is calculated using the formula: $$\text{scaled value} = \frac{\text{value} - \text{min value}}{\text{max value} - \text{min value}}.$$

2. Standardization: This technique transforms the features to have zero mean and unit variance. It is calculated using the formula: $$\text{standardized value} = \frac{\text{value} - \text{mean}}{\text{standard deviation}}.$$

These data scaling techniques were applied to the numeric features in the dataset to ensure that they have a consistent scale and distribution.
