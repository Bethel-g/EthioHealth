"""
Data Preprocessing Pipeline for EthioHealth-AI
Handles: missing data imputation, encoding, scaling, feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


class EthiopianEDPreprocessor:
    """Preprocess Ethiopian ED patient data for modeling."""

    def __init__(self):
        """Initialize preprocessor with scalers and encoders."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        self.feature_columns = None
        self.target_column = "los_hours"

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using:
        - Forward fill + mean imputation for vitals (common in LMICs)
        - Mode imputation for categorical
        """
        df = df.copy()

        # Vitals: median imputation by triage level (context-aware)
        vital_columns = [
            "systolic_bp", "diastolic_bp", "heart_rate",
            "respiratory_rate", "spo2", "temperature_c"
        ]

        for col in vital_columns:
            if col in df.columns and df[col].isnull().sum() > 0:
                # Impute by triage category to preserve clinical context
                df[col] = df.groupby("triage_category")[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Fill remaining with global median
                df[col].fillna(df[col].median(), inplace=True)

        # Categorical: mode imputation
        categorical_cols = ["chief_complaint", "disposition", "imaging_ordered", "lab_ordered"]
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown", inplace=True)

        print(f"✓ Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering:
        - Arrival time features (peak hours, day patterns)
        - Vital sign combinations (BP ratio, HR variability)
        - Complaint risk scores
        - Interaction features
        """
        df = df.copy()

        # ===== Arrival Time Features =====
        df["is_peak_hour_binary"] = (df["is_peak_hour"] == True).astype(int)
        df["is_weekend"] = (df["arrival_day_of_week"].isin([5, 6])).astype(int)
        df["is_night_shift"] = (df["arrival_hour"].isin([22, 23, 0, 1, 2, 3, 4, 5])).astype(int)

        # ===== Vital Sign Combinations =====
        # Mean arterial pressure (MAP)
        df["mean_arterial_pressure"] = (
            df["systolic_bp"] + 2 * df["diastolic_bp"]
        ) / 3
        # Pulse pressure
        df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
        # Vital stability score (lower = more abnormal)
        df["vital_stability_score"] = (
            (df["systolic_bp"].between(100, 140)).astype(int) +
            (df["heart_rate"].between(60, 110)).astype(int) +
            (df["respiratory_rate"].between(12, 20)).astype(int) +
            (df["spo2"] >= 95).astype(int)
        )

        # ===== Chief Complaint Risk Scores =====
        complaint_risk = {
            "Malaria": 3,
            "Trauma": 4,
            "Maternal emergency": 4,
            "Typhoid": 3,
            "Respiratory infection": 2,
            "Abdominal pain": 2,
            "Hypertensive emergency": 2,
            "Diarrheal disease": 1,
            "Other": 1
        }
        df["complaint_risk_score"] = df["chief_complaint"].map(
            complaint_risk
        ).fillna(1)

        # ===== Age Groups =====
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 5, 18, 35, 60, 100],
            labels=["Infant", "Child", "Adult", "Senior", "Elderly"]
        )

        # ===== Transport Mode Risk =====
        transport_risk = {
            "Walk": 1,
            "Taxi": 2,
            "Car": 2,
            "Ambulance": 4  # Already critical
        }
        df["transport_risk"] = df["transport_mode"].map(transport_risk).fillna(1)

        # ===== Resource Usage Features =====
        df["resource_intensity"] = (
            (df["imaging_ordered"] != "No").astype(int) +
            (df["lab_ordered"] != "No").astype(int) +
            df["medication_intensive"]
        )

        # ===== Interaction Features =====
        df["high_acuity_with_fever"] = (
            (df["triage_category"] <= 2).astype(int) *
            (df["temperature_c"] > 38.5).astype(int)
        )
        df["trauma_with_abnormal_vitals"] = (
            (df["chief_complaint"] == "Trauma").astype(int) *
            (df["vital_stability_score"] < 2).astype(int)
        )

        print(f"✓ Features engineered. Total features: {len(df.columns)}")
        return df

    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables using LabelEncoder."""
        df = df.copy()

        categorical_cols = [
            "gender", "region", "transport_mode", "chief_complaint",
            "disposition", "imaging_ordered", "lab_ordered", "age_group"
        ]

        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(
                        df[col].astype(str)
                    )
                else:
                    df[col] = self.label_encoders[col].transform(
                        df[col].astype(str)
                    )

        print(f"✓ Categorical variables encoded: {len(categorical_cols)} features")
        return df

    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns not needed for modeling."""
        df = df.copy()

        drop_cols = [
            "patient_id",  # ID column
            "arrival_time",  # Replaced by features
            "outcome_severity"  # Highly correlated with target
        ]

        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        print(f"✓ Dropped unnecessary columns: {len(drop_cols)} features")
        return df

    def prepare_dataset(self, df: pd.DataFrame, fit: bool = True) -> tuple:
        """
        Run complete preprocessing pipeline.

        Args:
            df: Input dataframe
            fit: Whether to fit scalers/encoders (True for training, False for new data)

        Returns:
            X: Feature matrix
            y: Target variable
        """
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60)

        # Step 1: Handle missing values
        df = self.handle_missing_values(df)

        # Step 2: Feature engineering
        df = self.engineer_features(df)

        # Step 3: Drop unnecessary columns
        df = self.drop_unnecessary_columns(df)

        # Step 4: Encode categorical variables
        df = self.encode_categorical(df, fit=fit)

        # Step 5: Separate features and target
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Store feature columns for later reference
        self.feature_columns = X.columns.tolist()

        # Step 6: Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)

        print(f"✓ Features scaled using StandardScaler")
        print(f"\nFinal dataset shape: X={X_scaled.shape}, y={y.shape}")
        print(f"Features: {', '.join(X_scaled.columns[:10])}... ({len(X_scaled.columns)} total)")
        print(f"Target distribution: μ={y.mean():.2f}h, σ={y.std():.2f}h, "
              f"min={y.min():.2f}h, max={y.max():.2f}h")

        return X_scaled, y


def main():
    """Example usage of preprocessing pipeline."""
    # Load synthetic data
    df = pd.read_csv("/home/betheln/projects/EthioHealth-AI/data/synthetic_ed_data.csv")

    # Initialize and run preprocessing
    preprocessor = EthiopianEDPreprocessor()
    X, y = preprocessor.prepare_dataset(df, fit=True)

    # Save preprocessed data
    X.to_csv("/home/betheln/projects/EthioHealth-AI/data/X_preprocessed.csv", index=False)
    y.to_csv("/home/betheln/projects/EthioHealth-AI/data/y_preprocessed.csv", index=False)

    print(f"\n✓ Preprocessed data saved")
    print(f"  - X: data/X_preprocessed.csv")
    print(f"  - y: data/y_preprocessed.csv")

    return preprocessor, X, y


if __name__ == "__main__":
    main()
