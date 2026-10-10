"""
SmartCart Customer Segmentation - Preprocessing & Feature Engineering Module
Author: Bhavesh Srivastava
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


class SmartCartPreprocessor:
    """
    Robust data preprocessing and domain-driven feature engineering pipeline
    for e-commerce customer transaction records.
    """

    def __init__(self, income_outlier_threshold: float = 600000.0, age_outlier_threshold: float = 90.0):
        self.income_outlier_threshold = income_outlier_threshold
        self.age_outlier_threshold = age_outlier_threshold

        # Transformers
        self.imputer = SimpleImputer(strategy="median")
        self.ohe = OneHotEncoder(sparse_output=False)
        self.ordinal_encoder = OrdinalEncoder(categories=[["Undergraduate", "Graduate", "Post Graduate"]])
        self.scaler = StandardScaler()

        self.fitted = False
        self.feature_names = None

    def fit_transform(self, df_raw: pd.DataFrame):
        """
        Fits all imputers, encoders, and scalers on raw customer data and transforms features.
        
        Args:
            df_raw (pd.DataFrame): Raw customer dataset.
            
        Returns:
            tuple: (X_scaled, df_cleaned, df_processed)
        """
        df = df_raw.copy()

        # 1. Median Imputation for missing Income (robust to heavy-tail skewness)
        df[["Income"]] = self.imputer.fit_transform(df[["Income"]])

        # 2. Temporal & Demographic Feature Engineering
        datetime_col = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
        current_year = datetime_col.dt.year.max()
        reference_date = datetime_col.max()

        df["Age"] = current_year - df["Year_Birth"]
        df["Customer_tenure"] = (reference_date - datetime_col).dt.days

        # 3. Monetary Aggregation (Total Spend across all 6 merchandise categories)
        spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        df["Total_Spending"] = df[spend_cols].sum(axis=1)

        # 4. Household Burden Metric
        df["Total_Kids"] = df["Kidhome"] + df["Teenhome"]

        # 5. Categorical Harmonization
        df["Education"] = df["Education"].map({
            "Basic": "Undergraduate",
            "2n Cycle": "Undergraduate",
            "Graduation": "Graduate",
            "PhD": "Post Graduate",
            "Master": "Post Graduate"
        })

        df["Living_with"] = df["Marital_Status"].replace({
            'Together': 'Partner',
            'Married': 'Partner',
            'Single': 'Alone',
            'Divorced': 'Alone',
            'Widow': 'Alone',
            'Absurd': 'Alone',
            'YOLO': 'Alone'
        })

        # 6. Drop redundant raw attributes
        cols_to_drop = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer"] + spend_cols
        df_cleaned = df.drop(columns=cols_to_drop)

        # 7. Statistical Outlier Filtering
        df_cleaned = df_cleaned[
            (df_cleaned["Income"] < self.income_outlier_threshold) & 
            (df_cleaned["Age"] < self.age_outlier_threshold)
        ]

        # 8. Categorical Encoding
        living_encoded = self.ohe.fit_transform(df_cleaned[["Living_with"]])
        living_df = pd.DataFrame(
            living_encoded, 
            columns=self.ohe.get_feature_names_out(), 
            index=df_cleaned.index
        )
        df_cleaned = pd.concat([df_cleaned, living_df], axis=1).drop(columns=["Living_with"])

        df_cleaned["Education"] = self.ordinal_encoder.fit_transform(df_cleaned[["Education"]])

        self.feature_names = df_cleaned.columns.tolist()

        # 9. Z-Score Standardization
        X_scaled = self.scaler.fit_transform(df_cleaned)
        self.fitted = True

        return X_scaled, df_cleaned
