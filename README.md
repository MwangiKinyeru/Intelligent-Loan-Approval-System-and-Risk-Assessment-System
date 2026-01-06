# Intelligent-Loan-Approval-System-and-Risk-Assessment-System
The system approves loans, determines loan eligibility, and calculates interest using a risk-based pricing model with the KESONIA formula.

##  Overview
PrimeTrust Loan Intelligence System a full-stack machine learning application that combines classification, regression, and clustering to automate loan decisions. Users submit loan applications through a web interface and receive instant approval status, approval probability, loan limits, interest rates, and repayment details. All decisions are logged for audit and monitoring and saved in google sheets.

## Features
>> * Loan Approval Prediction: Binary classification with approval probability
>> * Loan Amount Estimation: Predicts maximum eligible loan amount
>> * Risk Segmentation: Clusters borrowers into risk groups
>> * Dynamic Interest Rate Pricing: Risk-based interest rate assignment
>> * Web Application: Interactive loan application dashboard
>> * EMI & Repayment Plan Calculator
>> * Decision Logging: Automatic storage of all applications in Google Sheets

##  System Architecture
<p align="center">
  <img src="https://github.com/MwangiKinyeru/Intelligent-Loan-Approval-System-and-Risk-Assessment-System/blob/main/images/flow.png" />
</p>

## Technical Stack
>> * Backend: Python, Flask
>> * Machine Learning: scikit-learn, XGBoost
>> * Data Processing: Pandas, NumPy
>> * Model Persistence: Joblib, Pickle
>> * Frontend: HTML, CSS, JavaScript
>> * Deployment: Render
>> * Logging & Storage: Google Apps Script + Google Sheets

## How It Works
This system uses a two-stage decision process.

First, a classification model evaluates the borrower and returns an approval probability. If the applicant is rejected, the decision is logged out immediately but the details ares saved in the google sheets and values like available loan limit, interest rate and requested loan are filled as 0 . If approved, the system predicts a maximum loan limit and assigns a risk-based interest rate using borrower segmentation.

The user then selects a desired loan amount within their approved limit. The system calculates the optimal repayment tenure and monthly EMI, generates an approval summary, and logs the final loan decision into google sheet.

This approach reflects real-world lending workflows used by digital banks.

## User Interface Preview
<p float="left">
  <img src="https://github.com/MwangiKinyeru/Intelligent-Loan-Approval-System-and-Risk-Assessment-System/blob/main/images/Capture.PNG" width="45%" />
  <img src="https://github.com/MwangiKinyeru/Intelligent-Loan-Approval-System-and-Risk-Assessment-System/blob/main/images/Capture%201.PNG" width="45%" />
</p>

## Deployment

The Intelligent-Loan-Approval-System-and-Risk-Assessment-System was Deployed via render as a web application. To access the the finance bot click the link below
>>> [Deployment Link](https://customer-segmentation-g0b9.onrender.com)