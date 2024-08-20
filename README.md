# Heart Disease Prediction Project

# Overview
This project aims to predict the presence of heart disease in patients based on several medical and demographic attributes. The model is built using various machine learning algorithms, including Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN). The project compares these models to identify the most accurate algorithm for predicting heart disease.

# Dataset
The dataset used in this project consists of 270 rows and 14 columns, representing various attributes that are potentially indicative of heart disease. These attributes include:

1. **Age**: The age of the patient.
2. **Sex**: Gender of the patient (1 = male; 0 = female).
3. **Chest Pain Type**: Type of chest pain experienced by the patient (values from 1 to 4).
4. **Blood Pressure (BP)**: Resting blood pressure in mm Hg.
5. **Cholesterol**: Serum cholesterol in mg/dl.
6. **Fasting Blood Sugar (FBS)**: FBS over 120 mg/dl (1 = true; 0 = false).
7. **Electrocardiographic Results (EKG results)**: Resting electrocardiographic results (values 0 to 2).
8. **Thalach**: Maximum heart rate achieved.
9. **Exercise Angina**: Exercise-induced angina (1 = yes; 0 = no).
10. **Oldpeak**: ST depression induced by exercise relative to rest.
11. **Slope of ST**: The slope of the peak exercise ST segment (values 1 to 3).
12. **CA**: Number of major vessels (0-3) colored by fluoroscopy.
13. **Thallium**: A nuclear stress test result (values 3, 6, 7).
14. **Target**: Diagnosis of heart disease (1 = presence of heart disease, 0 = absence of heart disease).

# Data Preprocessing
The dataset was checked for null values and duplicates. No missing data was found, making it suitable for further analysis. The data was then split into training and testing sets for model evaluation.

# Data Analysis
Several visualizations were employed to understand the relationship between different features:

- **Correlation Matrix**: A heatmap was generated to visualize the correlation between various features and the target variable.
- **Feature Distribution**: Histograms were plotted to visualize the distribution of different features.
- **Balance Check**: A bar plot was used to check the balance of the target variable.

# Model Implementation
The following machine learning algorithms were implemented:

1. **Logistic Regression**:
   - Achieved an accuracy of 89% on the test data.
   - Confusion matrix and classification report were generated to evaluate the performance.

2. **Decision Tree**:
   - Achieved an accuracy of 83% on the test data.
   - The model was evaluated using cross-validation and the confusion matrix.

3. **Random Forest**:
   - Achieved an accuracy of 89% on the test data.
   - The model was trained with 500 estimators and evaluated with a confusion matrix.

4. **Support Vector Machine (SVM)**:
   - Achieved an accuracy of 87% on the test data.
   - The linear kernel was used, and the model was evaluated with a confusion matrix.

5. **K-Nearest Neighbors (KNN)**:
   - Achieved an accuracy of 65% on the test data.
   - The model was evaluated with a confusion matrix and classification report.

# Model Comparison
The performance of all models was compared based on their accuracy scores. The results showed that the **Random Forest** and **Logistic Regression** models performed the best, each achieving an accuracy of approximately 89%. However, when considering other metrics like precision, recall, and F1-score, **Decision Tree** and **SVM** also showed competitive results.

# Final Model
The final model chosen for deployment is the **Decision Tree Classifier**, which demonstrated the best balance between accuracy, interpretability, and computational efficiency.

# Test Cases
Two test cases were evaluated using the Decision Tree model:
- **Case 1**: Predicted the presence of heart disease.
- **Case 2**: Predicted the absence of heart disease.

# Conclusion
The project concludes that the Decision Tree algorithm is the most effective model for predicting heart disease based on the given dataset. However, Random Forest and Logistic Regression are also strong contenders and could be considered depending on specific application requirements.

# How to Run
1. Ensure you have Python installed with the necessary libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`).
2. Open the Jupyter Notebook `heartDiseaseprediction_mini_project_final.ipynb`.
3. Run the cells sequentially to load the data, preprocess, visualize, and build the models.
4. The final section of the notebook allows you to input new patient data to predict the presence of heart disease using the Decision Tree model.

# Requirements
- Python 3.6 or higher
- Jupyter Notebook
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

This README provides a comprehensive guide to understanding, running, and interpreting the heart disease prediction project.
