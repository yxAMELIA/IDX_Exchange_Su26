# IDX_Exchange_Su26
# California Property Close Price Prediction

## Project Overview
This project predicts California residential property close prices using CRMLS MLS data.

## Data Source
- CRMLS California MLS dataset
- Only Residential + SingleFamilyResidence properties are used

## Target
- ClosePrice (final transaction price)

## Important Note (Data Leakage)
ListPrice and OriginalListPrice are excluded from all models to prevent data leakage.

## Workflow
1. Exploratory Data Analysis (EDA)
2. Data preprocessing
3. Baseline model (Linear Regression)
4. Tree-based models (Random Forest / XGBoost)
5. Feature engineering

## How to run
```bash
pip install -r requirements.txt
