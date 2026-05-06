# Diametrics

## Overview
This is a diabetes risk predicition tool that uses CDC data to predict diabetes risk using s logistic regression model.
Three models were compared based on f1 score: logistic regression, SVM and Naive Bayes. Logistic regression got the highest f1 score of the three models.

## Input for LR.ipynb, Naive_Bayes.ipynb or svm_model.ipynb
1. data_smote.csv (SMOTE data set)
2. data_undersampling.csv (Under sampled dataset)
3. data_oversampling.csv (Over sampled dataset)

## Set Up to run Streamlit
cd DiametricsFinal

streamlit run Home.py


