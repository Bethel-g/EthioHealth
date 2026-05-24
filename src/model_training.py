# """
# Model Training for EthioHealth-AI
# Trains multiple regression models to predict Length of Stay (LOS)
# Models: Linear Regression, Random Forest, XGBoost
# """

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib
# import warnings

# warnings.filterwarnings('ignore')


# class EthiopianEDModelTrainer:
#     """Train and evaluate regression models for LOS prediction."""

#     def __init__(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
#         """
#         Initialize trainer.

#         Args:
#             X: Feature matrix (preprocessed)
#             y: Target variable (LOS in hours)
#             test_size: Proportion for test set
#             val_size: Proportion for validation set (from training data)
#             random_state: For reproducibility
#         """
#         self.X = X
#         self.y = y
#         self.random_state = random_state

#         # Split: 60% train, 20% val, 20% test
#         X_temp, self.X_test, y_temp, self.y_test = train_test_split(
#             X, y, test_size=test_size, random_state=random_state
#         )

#         val_split = val_size / (1 - test_size)  # Adjust for remaining data
#         self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
#             X_temp, y_temp, test_size=val_split, random_state=random_state
#         )

#         print(f"\nData Split:")
#         print(f"  Train: {len(self.X_train)} samples")
#         print(f"  Val:   {len(self.X_val)} samples")
#         print(f"  Test:  {len(self.X_test)} samples")

#         self.models = {}
#         self.results = {}

#     @staticmethod
#     def calculate_metrics(y_true, y_pred, model_name="Model"):
#         """
#         Calculate regression metrics.

#         Args:
#             y_true: Ground truth values
#             y_pred: Predictions
#             model_name: For reporting

#         Returns:
#             Dictionary of metrics
#         """
#         mae = mean_absolute_error(y_true, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#         r2 = r2_score(y_true, y_pred)
#         mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#         return {
#             "MAE": mae,
#             "RMSE": rmse,
#             "R2": r2,
#             "MAPE": mape
#         }

#     def train_linear_regression(self):
#         """Train Linear Regression model."""
#         print("\n" + "="*60)
#         print("TRAINING: Linear Regression")
#         print("="*60)

#         model = LinearRegression()
#         model.fit(self.X_train, self.y_train)

#         # Predictions
#         y_pred_train = model.predict(self.X_train)
#         y_pred_val = model.predict(self.X_val)
#         y_pred_test = model.predict(self.X_test)

#         # Metrics
#         train_metrics = self.calculate_metrics(self.y_train, y_pred_train)
#         val_metrics = self.calculate_metrics(self.y_val, y_pred_val)
#         test_metrics = self.calculate_metrics(self.y_test, y_pred_test)

#         # Cross-validation
#         cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, 
#                                    scoring='r2', n_jobs=-1)

#         # Store
#         self.models["Linear Regression"] = model
#         self.results["Linear Regression"] = {
#             "train": train_metrics,
#             "val": val_metrics,
#             "test": test_metrics,
#             "cv_r2_mean": cv_scores.mean(),
#             "cv_r2_std": cv_scores.std(),
#             "predictions_test": y_pred_test
#         }

#         # Print results
#         print(f"\nTrain Metrics:")
#         for metric, value in train_metrics.items():
#             print(f"  {metric}: {value:.4f}")

#         print(f"\nValidation Metrics:")
#         for metric, value in val_metrics.items():
#             print(f"  {metric}: {value:.4f}")

#         print(f"\nTest Metrics:")
#         for metric, value in test_metrics.items():
#             print(f"  {metric}: {value:.4f}")

#         print(f"\nCross-Validation (5-fold):")
#         print(f"  R² Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

#         return model

#     def train_random_forest(self):
#         """Train Random Forest Regressor."""
#         print("\n" + "="*60)
#         print("TRAINING: Random Forest Regressor")
#         print("="*60)

#         model = RandomForestRegressor(
#             n_estimators=200,
#             max_depth=20,
#             min_samples_split=5,
#             min_samples_leaf=2,
#             random_state=self.random_state,
#             n_jobs=-1,
#             verbose=0
#         )
#         model.fit(self.X_train, self.y_train)

#         # Predictions
#         y_pred_train = model.predict(self.X_train)
#         y_pred_val = model.predict(self.X_val)
#         y_pred_test = model.predict(self.X_test)

#         # Metrics
#         train_metrics = self.calculate_metrics(self.y_train, y_pred_train)
#         val_metrics = self.calculate_metrics(self.y_val, y_pred_val)
#         test_metrics = self.calculate_metrics(self.y_test, y_pred_test)

#         # Cross-validation
#         cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5,
#                                    scoring='r2', n_jobs=-1)

#         # Store
#         self.models["Random Forest"] = model
#         self.results["Random Forest"] = {
#             "train": train_metrics,
#             "val": val_metrics,
#             "test": test_metrics,
#             "cv_r2_mean": cv_scores.mean(),
#             "cv_r2_std": cv_scores.std(),
#             "predictions_test": y_pred_test,
#             "feature_importance": model.feature_importances_
#         }

#         # Print results
#         print(f"\nTrain Metrics:")
#         for metric, value in train_metrics.items():
#             print(f"  {metric}: {value:.4f}")

#         print(f"\nValidation Metrics:")
#         for metric, value in val_metrics.items():
#             print(f"  {metric}: {value:.4f}")

#         print(f"\nTest Metrics:")
#         for metric, value in test_metrics.items():
#             print(f"  {metric}: {value:.4f}")

#         print(f"\nCross-Validation (5-fold):")
#         print(f"  R² Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

#         return model

#     def train_xgboost(self):
#         """Train XGBoost Regressor."""
#         print("\n" + "="*60)
#         print("TRAINING: XGBoost Regressor")
#         print("="*60)

#         model = XGBRegressor(
#             n_estimators=200,
#             max_depth=8,
#             learning_rate=0.05,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=self.random_state,
#             tree_method='hist',
#             device='cpu',
#             verbosity=0
#         )
#         model.fit(
#             self.X_train, self.y_train,
#             eval_set=[(self.X_val, self.y_val)],
#             early_stopping_rounds=20,
#             verbose=False
#         )

#         # Predictions
#         y_pred_train = model.predict(self.X_train)
#         y_pred_val = model.predict(self.X_val)
#         y_pred_test = model.predict(self.X_test)

#         # Metrics
#         train_metrics = self.calculate_metrics(self.y_train, y_pred_train)
#         val_metrics = self.calculate_metrics(self.y_val, y_pred_val)
#         test_metrics = self.calculate_metrics(self.y_test, y_pred_test)

#         # Cross-validation
#         cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5,
#                                    scoring='r2', n_jobs=-1)

#         # Store
#         self.models["XGBoost"] = model
#         self.results["XGBoost"] = {
#             "train": train_metrics,
#             "val": val_metrics,
#             "test": test_metrics,
#             "cv_r2_mean": cv_scores.mean(),
#             "cv_r2_std": cv_scores.std(),
#             "predictions_test": y_pred_test,
#             "feature_importance": model.feature_importances_
#         }

#         # Print results
#         print(f"\nTrain Metrics:")
#         for metric, value in train_metrics.items():
#             print(f"  {metric}: {value:.4f}")

#         print(f"\nValidation Metrics:")
#         for metric, value in val_metrics.items():
#             print(f"  {metric}: {value:.4f}")

#         print(f"\nTest Metrics:")
#         for metric, value in test_metrics.items():
#             print(f"  {metric}: {value:.4f}")

#         print(f"\nCross-Validation (5-fold):")
#         print(f"  R² Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

#         return model

#     def print_model_comparison(self):
#         """Compare all trained models."""
#         print("\n" + "="*70)
#         print("MODEL COMPARISON - TEST SET PERFORMANCE")
#         print("="*70)

#         comparison_df = pd.DataFrame()

#         for model_name, result in self.results.items():
#             test_metrics = result["test"]
#             comparison_df = pd.concat([comparison_df, pd.DataFrame({
#                 "Model": [model_name],
#                 "MAE (h)": [test_metrics["MAE"]],
#                 "RMSE (h)": [test_metrics["RMSE"]],
#                 "R²": [test_metrics["R2"]],
#                 "MAPE (%)": [test_metrics["MAPE"]],
#                 "CV R² (mean)": [result["cv_r2_mean"]]
#             })], ignore_index=True)

#         comparison_df = comparison_df.sort_values("R²", ascending=False).reset_index(drop=True)
#         print("\n" + comparison_df.to_string(index=False))

#         # Clinical interpretation
#         print("\n" + "="*70)
#         print("CLINICAL INTERPRETATION")
#         print("="*70)
#         best_model_name = comparison_df.iloc[0]["Model"]
#         best_mae = comparison_df.iloc[0]["MAE (h)"]
#         best_r2 = comparison_df.iloc[0]["R²"]

#         print(f"\nBest Performing Model: {best_model_name}")
#         print(f"  - Prediction Error (MAE): ±{best_mae:.2f} hours")
#         print(f"  - Model explains {best_r2*100:.1f}% of variance in LOS")
#         print(f"\n  Clinical Significance:")
#         print(f"    - For a patient with predicted LOS of 8 hours:")
#         print(f"      Actual LOS likely: {8-best_mae:.1f} to {8+best_mae:.1f} hours")
#         print(f"    - This allows ED staff to allocate beds with ~{best_mae*60:.0f} min error")

#         return comparison_df

#     def train_all_models(self):
#         """Train all three models."""
#         print("\n" + "="*70)
#         print("ETHIOHEALTH-AI: MODEL TRAINING PIPELINE")
#         print("="*70)

#         self.train_linear_regression()
#         self.train_random_forest()
#         self.train_xgboost()

#         comparison = self.print_model_comparison()

#         return comparison

#     def save_models(self, model_dir="/home/betheln/projects/EthioHealth-AI/models"):
#         """Save trained models to disk."""
#         for model_name, model in self.models.items():
#             filepath = f"{model_dir}/{model_name.replace(' ', '_').lower()}_model.pkl"
#             joblib.dump(model, filepath)
#             print(f"✓ Saved: {filepath}")

#     def get_best_model(self):
#         """Return best model based on R² score."""
#         best_model_name = max(
#             self.results.items(),
#             key=lambda x: x[1]["test"]["R2"]
#         )[0]
#         return best_model_name, self.models[best_model_name]


# # def main():
# #     """Train models on preprocessed data."""
# #     # Load preprocessed data
# #     X = pd.read_csv("/home/betheln/projects/EthioHealth-AI/data/X_preprocessed.csv")
# #     y = pd.read_csv("/home/betheln/projects/EthioHealth-AI/data/y_preprocessed.csv").squeeze()

# #     # Train models
# #     trainer = EthiopianEDModelTrainer(X, y)
# #     trainer.train_all_models()

# #     # Save models
# #     trainer.save_models()

# #     # Get best model
# #     best_name, best_model = trainer.get_best_model()
# #     print(f"\n✓ Best model: {best_name}")



# def main():
#     print("🚀 Starting training pipeline...")

#     X = pd.read_csv("/home/betheln/projects/EthioHealth-AI/data/X_preprocessed.csv")
#     y = pd.read_csv("/home/betheln/projects/EthioHealth-AI/data/y_preprocessed.csv").squeeze()

#     print("✅ Data loaded")

#     trainer = EthiopianEDModelTrainer(X, y)

#     print("🚀 Training models...")
#     trainer.train_all_models()

#     print("💾 Saving models...")
#     trainer.save_models()

#     print("🎯 Getting best model...")
#     best_name, _ = trainer.get_best_model()

#     print(f"✅ DONE - Best model: {best_name}")

#     return trainer


# if __name__ == "__main__":
#     main()




"""
Model Training for EthioHealth-AI
Trains regression models to predict Length of Stay (LOS)
"""

import pandas as pd
import numpy as np
import os
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings('ignore')


class EthiopianEDModelTrainer:
    """Train and evaluate LOS prediction models."""

    def __init__(self, X, y, test_size=0.2, val_size=0.2, random_state=42):

        self.X = X
        self.y = y
        self.random_state = random_state

        # Split dataset
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        val_split = val_size / (1 - test_size)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_split, random_state=random_state
        )

        print("\nData Split:")
        print(f"  Train: {len(self.X_train)}")
        print(f"  Val:   {len(self.X_val)}")
        print(f"  Test:  {len(self.X_test)}")

        self.models = {}
        self.results = {}

    # =========================
    # Metrics
    # =========================
    def calculate_metrics(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape
        }

    # =========================
    # Linear Regression
    # =========================
    def train_linear_regression(self):
        print("\nTraining Linear Regression...")

        model = LinearRegression()
        model.fit(self.X_train, self.y_train)

        self._evaluate_model("Linear Regression", model)
        return model

    # =========================
    # Random Forest
    # =========================
    def train_random_forest(self):
        print("\nTraining Random Forest...")

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            random_state=self.random_state,
            n_jobs=-1
        )

        model.fit(self.X_train, self.y_train)

        self._evaluate_model("Random Forest", model)
        return model

    # =========================
    # XGBoost (FIXED)
    # =========================
    def train_xgboost(self):
        print("\nTraining XGBoost...")

        model = XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            tree_method="hist",
            verbosity=0
        )

        # ⚠ FIX: NO early_stopping_rounds (new xgboost compatibility)
        model.fit(self.X_train, self.y_train)

        self._evaluate_model("XGBoost", model)
        return model

    # =========================
    # Shared evaluation
    # =========================
    def _evaluate_model(self, name, model):

        y_train_pred = model.predict(self.X_train)
        y_val_pred = model.predict(self.X_val)
        y_test_pred = model.predict(self.X_test)

        train_metrics = self.calculate_metrics(self.y_train, y_train_pred)
        val_metrics = self.calculate_metrics(self.y_val, y_val_pred)
        test_metrics = self.calculate_metrics(self.y_test, y_test_pred)

        cv_scores = cross_val_score(
            model,
            self.X_train,
            self.y_train,
            cv=5,
            scoring="r2"
        )

        self.models[name] = model
        self.results[name] = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "preds": y_test_pred
        }

        print(f"\n{name} Results:")
        print(f"  MAE: {test_metrics['MAE']:.3f}")
        print(f"  RMSE: {test_metrics['RMSE']:.3f}")
        print(f"  R2: {test_metrics['R2']:.3f}")

    # =========================
    # Train all models
    # =========================
    def train_all_models(self):
        print("\n==============================")
        print("ETHIOHEALTH-AI TRAINING")
        print("==============================")

        self.train_linear_regression()
        self.train_random_forest()
        self.train_xgboost()

        return self.results

    # =========================
    # Save models (FIXED)
    # =========================
    def save_models(self, model_dir="models"):

        os.makedirs(model_dir, exist_ok=True)

        for name, model in self.models.items():
            filename = name.lower().replace(" ", "_") + "_model.pkl"
            path = os.path.join(model_dir, filename)

            joblib.dump(model, path)
            print(f"Saved: {path}")

    # =========================
    # Best model
    # =========================
    def get_best_model(self):

        best = max(
            self.results.items(),
            key=lambda x: x[1]["test"]["R2"]
        )

        return best[0], self.models[best[0]]


# =========================
# MAIN
# =========================
def main():

    print("🚀 Starting training pipeline...")

    X = pd.read_csv("data/X_preprocessed.csv")
    y = pd.read_csv("data/y_preprocessed.csv").squeeze()

    print("✅ Data loaded")

    trainer = EthiopianEDModelTrainer(X, y)

    trainer.train_all_models()

    print("\n💾 Saving models...")
    trainer.save_models()

    best_name, _ = trainer.get_best_model()
    print(f"\n🏆 Best model: {best_name}")

    print("\n✅ TRAINING COMPLETE")


if __name__ == "__main__":
    main()