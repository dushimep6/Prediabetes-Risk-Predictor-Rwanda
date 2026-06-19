# Prediabetes Risk Predictor — Rwanda

A bilingual self-assessment decision support tool for prediabetes risk prediction among Rwandan adults, developed using the 2021–2022 Rwanda WHO STEPS survey data.

## About

This tool uses a Logistic Regression model (AUC=0.9885) trained on 5,010 Rwandan adults to predict individual prediabetes risk. It classifies users into:

- 🟢 **Low Risk** — probability < 20%
- 🟡 **Moderate Risk** — probability 20–58.75%
- 🔴 **High Risk** — probability ≥ 58.75%

## Features

- Available in **English** and **Kinyarwanda**
- Tab 1: Basic assessment (no equipment needed)
- Tab 2: Optional clinical measurements (waist circumference, cholesterol)
- Personalised lifestyle recommendations per risk zone

## Study

**Title:** Development and Evaluation of Machine Learning Models for Prediabetes Risk Prediction Among Rwandan Adults and Translation into a Bilingual Self-Assessment Decision Support Tool

**Author:** Pasteur DUSHIMIMANA  
**Institution:** University of Rwanda, College of Medicine and Health Sciences  
**Supervisor:** Assoc. Prof. Gerard RUSHINGABIGWI

## Data

Rwanda WHO STEPwise Approach to Surveillance (STEPS) Survey 2021–2022  
IRB Approval: CMHS/IRB/349/2026

## Usage

Enter your health information in Tab 1. Optionally add measurements in Tab 2 for a more precise estimate. Click **Calculate Risk** to see your result and personalised recommendations.
