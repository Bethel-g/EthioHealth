"""
Ethiopian ED Synthetic Data Generator
Generates realistic patient records reflecting Ethiopian referral hospital context:
- High malaria/typhoid burden
- Trauma from road accidents
- Late presenters (advanced disease)
- High missing data rates (20-40% vitals missing)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class EthiopianEDDataGenerator:
    """Generate synthetic Ethiopian ED patient data."""

    def __init__(self, n_samples=10000, random_state=42):
        """
        Initialize the data generator.

        Args:
            n_samples (int): Number of synthetic patient records to generate
            random_state (int): For reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)

        # Ethiopian-specific parameters
        self.regions = [
            "Addis Ababa", "Oromia", "SNNP", "Amhara",
            "Tigray", "Benishangul", "Gambella", "Harari"
        ]
        self.complaints = {
            "Malaria": 0.25,  # High burden
            "Typhoid": 0.12,
            "Trauma": 0.18,  # Road accidents
            "Abdominal pain": 0.10,
            "Respiratory infection": 0.12,
            "Maternal emergency": 0.08,
            "Diarrheal disease": 0.07,
            "Hypertensive emergency": 0.05,
            "Other": 0.03
        }
        self.transport_modes = ["Walk", "Car", "Ambulance", "Taxi"]

    def generate_demographics(self) -> pd.DataFrame:
        """Generate patient demographics."""
        age_mu, age_sigma = 35, 25
        ages = np.abs(np.random.normal(age_mu, age_sigma, self.n_samples))
        ages = np.clip(ages, 1, 95)

        gender = np.random.choice(["M", "F"], self.n_samples, p=[0.55, 0.45])
        region = np.random.choice(self.regions, self.n_samples)
        transport = np.random.choice(
            self.transport_modes, self.n_samples, p=[0.40, 0.35, 0.15, 0.10]
        )

        return pd.DataFrame({
            "age": ages,
            "gender": gender,
            "region": region,
            "transport_mode": transport
        })

    def generate_arrival_times(self) -> pd.DataFrame:
        """Generate arrival times (peak hours: 8-11am, 6-10pm)."""
        base_date = datetime(2023, 1, 1)
        hours = []
        for _ in range(self.n_samples):
            # Peak hours more likely
            if np.random.random() < 0.6:  # 60% during peak hours
                if np.random.random() < 0.5:
                    hour = np.random.choice(range(8, 12))  # Morning peak
                else:
                    hour = np.random.choice(range(18, 23))  # Evening peak
            else:
                hour = np.random.randint(0, 24)

            minute = np.random.randint(0, 60)
            hours.append(hour)

        arrival_times = [base_date + timedelta(hours=int(h), minutes=int(m))
                        for h, m in zip(hours, np.random.randint(0, 60, self.n_samples))]

        df = pd.DataFrame({
            "arrival_time": arrival_times,
            "arrival_hour": hours,
            "arrival_day_of_week": [t.weekday() for t in arrival_times],
            "arrival_month": [t.month for t in arrival_times],
            "is_peak_hour": [(h >= 8 and h <= 11) or (h >= 18 and h <= 22) for h in hours]
        })
        return df

    def generate_chief_complaints(self) -> pd.DataFrame:
        """Generate chief complaints (Ethiopian context)."""
        complaints = np.random.choice(
            list(self.complaints.keys()),
            self.n_samples,
            p=list(self.complaints.values())
        )
        return pd.DataFrame({"chief_complaint": complaints})

    def generate_vital_signs(self, complaints: np.ndarray) -> pd.DataFrame:
        """
        Generate vital signs with Ethiopian context:
        - High missing data (20-40%)
        - Variation based on complaint type
        """
        missing_rate = np.random.uniform(0.20, 0.40)

        # Systolic BP (usually elevated in Ethiopian ED due to late presentation)
        systolic_bp = np.random.normal(130, 25, self.n_samples)
        systolic_bp = np.clip(systolic_bp, 80, 220)

        # Diastolic BP
        diastolic_bp = np.random.normal(80, 15, self.n_samples)
        diastolic_bp = np.clip(diastolic_bp, 40, 130)

        # Heart rate (elevated in malaria/sepsis)
        hr = np.random.normal(95, 20, self.n_samples)
        hr = np.clip(hr, 40, 180)

        # Respiratory rate
        rr = np.random.normal(20, 5, self.n_samples)
        rr = np.clip(rr, 10, 40)

        # SpO2 (often low in respiratory cases)
        spo2 = np.random.normal(96, 4, self.n_samples)
        spo2 = np.clip(spo2, 75, 100)

        # Temperature (fever in malaria/infection)
        temp_c = np.random.normal(37.5, 1.2, self.n_samples)
        temp_c = np.clip(temp_c, 35, 41)

        # Apply complaint-specific adjustments
        for i, complaint in enumerate(complaints):
            if complaint == "Malaria":
                temp_c[i] += np.random.normal(1.0, 0.5)  # Higher fever
                rr[i] += np.random.normal(2, 1)
            elif complaint == "Respiratory infection":
                rr[i] += np.random.normal(3, 1)
                spo2[i] -= np.random.normal(3, 2)
            elif complaint == "Trauma":
                hr[i] += np.random.normal(10, 5)
                systolic_bp[i] -= np.random.normal(5, 10)

        # Apply missing data pattern (MCAR - Missing Completely At Random)
        mask = np.random.random((self.n_samples, 6)) < missing_rate

        vitals = pd.DataFrame({
            "systolic_bp": np.where(mask[:, 0], np.nan, systolic_bp),
            "diastolic_bp": np.where(mask[:, 1], np.nan, diastolic_bp),
            "heart_rate": np.where(mask[:, 2], np.nan, hr),
            "respiratory_rate": np.where(mask[:, 3], np.nan, rr),
            "spo2": np.where(mask[:, 4], np.nan, spo2),
            "temperature_c": np.where(mask[:, 5], np.nan, temp_c)
        })

        return vitals

    def generate_triage_category(self, vitals: pd.DataFrame, complaints: np.ndarray) -> np.ndarray:
        """
        Assign triage category (Canadian Acuity Scale adapted for Ethiopia).
        Categories: 1=Resuscitation, 2=Emergent, 3=Urgent, 4=Semi-urgent, 5=Non-urgent
        """
        triage = np.full(self.n_samples, 3, dtype=int)  # Default: urgent

        for i in range(self.n_samples):
            # Resuscitation
            if (vitals.loc[i, "systolic_bp"] < 90 or
                vitals.loc[i, "heart_rate"] > 140 or
                vitals.loc[i, "respiratory_rate"] > 35):
                triage[i] = 1
            # Emergent
            elif (vitals.loc[i, "systolic_bp"] < 100 or
                  vitals.loc[i, "spo2"] < 90 or
                  complaints[i] in ["Maternal emergency", "Trauma"]):
                triage[i] = 2
            # Semi-urgent
            elif complaints[i] in ["Abdominal pain", "Diarrheal disease"]:
                triage[i] = 4
            # Non-urgent
            elif complaints[i] == "Other":
                triage[i] = 5

        return triage

    def generate_outcomes(self, triage: np.ndarray, complaints: np.ndarray,
                         vitals: pd.DataFrame) -> tuple:
        """
        Generate LOS (target), disposition, and complications.
        LOS varies by triage and complaint (Ethiopian context: late presentations = longer stay)
        """
        los_hours = np.zeros(self.n_samples)
        disposition = []
        outcome_severity = np.zeros(self.n_samples)

        for i in range(self.n_samples):
            # Base LOS by triage
            if triage[i] == 1:
                los_hours[i] = np.random.normal(24, 8)  # Resuscitation: ~24h
            elif triage[i] == 2:
                los_hours[i] = np.random.normal(16, 6)  # Emergent: ~16h
            elif triage[i] == 3:
                los_hours[i] = np.random.normal(8, 4)   # Urgent: ~8h
            elif triage[i] == 4:
                los_hours[i] = np.random.normal(4, 2)   # Semi-urgent: ~4h
            else:
                los_hours[i] = np.random.normal(2, 1)   # Non-urgent: ~2h

            # Complaint-specific adjustments
            if complaints[i] == "Malaria":
                los_hours[i] += np.random.normal(3, 2)  # Admission likely
            elif complaints[i] == "Trauma":
                los_hours[i] += np.random.normal(5, 3)
            elif complaints[i] == "Maternal emergency":
                los_hours[i] += np.random.normal(8, 4)

            # Enforce minimum LOS
            los_hours[i] = max(los_hours[i], 0.5)

            # Disposition
            if triage[i] in [1, 2]:
                disposition.append(np.random.choice(
                    ["Admission", "ICU"], p=[0.60, 0.40]))
            elif triage[i] == 3:
                disposition.append(np.random.choice(
                    ["Admission", "Discharge"], p=[0.50, 0.50]))
            else:
                disposition.append(np.random.choice(
                    ["Discharge", "Admission"], p=[0.80, 0.20]))

            # Outcome severity (1-5 scale)
            outcome_severity[i] = triage[i]
            if disposition[i] == "ICU":
                outcome_severity[i] = min(outcome_severity[i] + 1, 5)

        return los_hours, disposition, outcome_severity

    def generate_resource_usage(self, triage: np.ndarray, los_hours: np.ndarray) -> pd.DataFrame:
        """Generate ED resource usage estimates."""
        resources = pd.DataFrame({
            "imaging_ordered": np.random.choice(
                ["No", "X-ray", "Ultrasound", "CT"], self.n_samples,
                p=[0.40, 0.30, 0.20, 0.10]
            ),
            "lab_ordered": np.random.choice(
                ["No", "Basic", "Extended"], self.n_samples,
                p=[0.30, 0.50, 0.20]
            ),
            "medication_intensive": (triage <= 2).astype(int),
            "staff_hours_estimate": los_hours * np.random.uniform(0.3, 0.7, self.n_samples)
        })
        return resources

    def generate_dataset(self) -> pd.DataFrame:
        """Generate complete synthetic dataset."""
        print(f"Generating {self.n_samples} Ethiopian ED patient records...")

        # Generate components
        demographics = self.generate_demographics()
        arrivals = self.generate_arrival_times()
        complaints_df = self.generate_chief_complaints()
        vitals = self.generate_vital_signs(complaints_df["chief_complaint"].values)

        # Generate outcomes based on vitals and complaints
        triage = self.generate_triage_category(
            vitals, complaints_df["chief_complaint"].values
        )
        los_hours, disposition, outcome_severity = self.generate_outcomes(
            triage, complaints_df["chief_complaint"].values, vitals
        )
        resources = self.generate_resource_usage(triage, los_hours)

        # Combine all
        df = pd.concat([
            demographics, arrivals, complaints_df, vitals,
            pd.DataFrame({"triage_category": triage}),
            pd.DataFrame({
                "los_hours": los_hours,
                "disposition": disposition,
                "outcome_severity": outcome_severity
            }),
            resources
        ], axis=1)

        # Add wait time (usually 2-6 hours before physician in Ethiopian context)
        df["wait_time_before_physician_hours"] = np.random.exponential(3, self.n_samples)
        df["wait_time_before_physician_hours"] = np.clip(
            df["wait_time_before_physician_hours"], 0.1, 12
        )

        # Add patient ID
        df.insert(0, "patient_id", [f"ETH_{str(i).zfill(6)}" for i in range(self.n_samples)])

        print(f"✓ Generated {len(df)} records")
        print(f"✓ Missing data rate: {(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%")

        return df


def main():
    """Generate and save synthetic dataset."""
    generator = EthiopianEDDataGenerator(n_samples=10000, random_state=42)
    df = generator.generate_dataset()

    # Save dataset
    df.to_csv("/home/betheln/projects/EthioHealth-AI/data/synthetic_ed_data.csv", index=False)
    print(f"\n✓ Dataset saved to: data/synthetic_ed_data.csv")

    # Print summary
    print(f"\nDataset Summary:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")

    return df


if __name__ == "__main__":
    main()
