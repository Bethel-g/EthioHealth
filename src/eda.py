"""
Exploratory Data Analysis for EthioHealth-AI
Visualizes patient characteristics, LOS patterns, mortality by wait time, correlations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EthiopianEDAnalyzer:
    """Perform EDA on Ethiopian ED data."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer.

        Args:
            df: Raw synthetic ED dataframe
        """
        self.df = df.copy()
        self.fig_dir = "/home/betheln/projects/EthioHealth-AI/notebooks/figures"

    def plot_los_distribution(self, save=True):
        """Visualize Length of Stay distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(self.df["los_hours"], bins=50, color="steelblue", alpha=0.7, edgecolor="black")
        axes[0].set_xlabel("Length of Stay (hours)", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Frequency", fontsize=11, fontweight="bold")
        axes[0].set_title("Distribution of ED Length of Stay", fontsize=12, fontweight="bold")
        axes[0].axvline(self.df["los_hours"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {self.df['los_hours'].mean():.1f}h")
        axes[0].axvline(self.df["los_hours"].median(), color="green", linestyle="--", linewidth=2, label=f"Median: {self.df['los_hours'].median():.1f}h")
        axes[0].legend()

        # Box plot by triage
        self.df.boxplot(column="los_hours", by="triage_category", ax=axes[1])
        axes[1].set_xlabel("Triage Category (1=Critical, 5=Non-urgent)", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Length of Stay (hours)", fontsize=11, fontweight="bold")
        axes[1].set_title("LOS by Triage Severity", fontsize=12, fontweight="bold")
        plt.suptitle("")

        plt.tight_layout()
        if save:
            plt.savefig(f"{self.fig_dir}/01_los_distribution.png", dpi=300, bbox_inches='tight')
            print("✓ Saved: 01_los_distribution.png")
        plt.show()

    def plot_peak_hours_analysis(self, save=True):
        """Analyze ED patient arrivals by hour (peak hours: 8-11am, 6-10pm)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Arrivals by hour
        hourly_arrivals = self.df["arrival_hour"].value_counts().sort_index()
        axes[0].bar(hourly_arrivals.index, hourly_arrivals.values, color="coral", alpha=0.7, edgecolor="black")
        axes[0].axvspan(8, 11, alpha=0.2, color="red", label="Morning peak")
        axes[0].axvspan(18, 22, alpha=0.2, color="orange", label="Evening peak")
        axes[0].set_xlabel("Hour of Day", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Number of Arrivals", fontsize=11, fontweight="bold")
        axes[0].set_title("ED Patient Arrivals by Hour", fontsize=12, fontweight="bold")
        axes[0].legend()
        axes[0].set_xticks(range(0, 24, 2))

        # LOS by peak vs off-peak
        peak_los = self.df[self.df["is_peak_hour"] == True]["los_hours"]
        offpeak_los = self.df[self.df["is_peak_hour"] == False]["los_hours"]
        axes[1].violinplot([peak_los, offpeak_los], positions=[1, 2], showmeans=True, showmedians=True)
        axes[1].set_xticks([1, 2])
        axes[1].set_xticklabels(["Peak Hours", "Off-Peak"])
        axes[1].set_ylabel("Length of Stay (hours)", fontsize=11, fontweight="bold")
        axes[1].set_title("LOS: Peak Hours vs Off-Peak", fontsize=12, fontweight="bold")

        plt.tight_layout()
        if save:
            plt.savefig(f"{self.fig_dir}/02_peak_hours_analysis.png", dpi=300, bbox_inches='tight')
            print("✓ Saved: 02_peak_hours_analysis.png")
        plt.show()

    def plot_complaint_analysis(self, save=True):
        """Analyze chief complaints and their outcomes."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Complaint frequency
        complaint_counts = self.df["chief_complaint"].value_counts()
        axes[0, 0].barh(complaint_counts.index, complaint_counts.values, color="skyblue", edgecolor="black")
        axes[0, 0].set_xlabel("Number of Patients", fontsize=11, fontweight="bold")
        axes[0, 0].set_title("Chief Complaints in Ethiopian ED", fontsize=12, fontweight="bold")

        # Mean LOS by complaint
        los_by_complaint = self.df.groupby("chief_complaint")["los_hours"].mean().sort_values()
        axes[0, 1].barh(los_by_complaint.index, los_by_complaint.values, color="lightcoral", edgecolor="black")
        axes[0, 1].set_xlabel("Mean LOS (hours)", fontsize=11, fontweight="bold")
        axes[0, 1].set_title("Average ED Stay by Chief Complaint", fontsize=12, fontweight="bold")

        # Triage distribution by complaint (top 5)
        top_complaints = self.df["chief_complaint"].value_counts().head(5).index
        triage_complaint = pd.crosstab(
            self.df[self.df["chief_complaint"].isin(top_complaints)]["chief_complaint"],
            self.df[self.df["chief_complaint"].isin(top_complaints)]["triage_category"]
        )
        triage_complaint.plot(kind="bar", ax=axes[1, 0], stacked=True, colormap="RdYlGn_r")
        axes[1, 0].set_ylabel("Number of Patients", fontsize=11, fontweight="bold")
        axes[1, 0].set_xlabel("Chief Complaint", fontsize=11, fontweight="bold")
        axes[1, 0].set_title("Triage Distribution by Top Complaints", fontsize=12, fontweight="bold")
        axes[1, 0].legend(title="Triage", labels=["Critical", "Emergent", "Urgent", "Semi-urgent", "Non-urgent"], loc="upper right")
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Disposition by complaint
        top_complaints = self.df["chief_complaint"].value_counts().head(5).index
        disposition_complaint = pd.crosstab(
            self.df[self.df["chief_complaint"].isin(top_complaints)]["chief_complaint"],
            self.df[self.df["chief_complaint"].isin(top_complaints)]["disposition"]
        )
        disposition_complaint.plot(kind="bar", ax=axes[1, 1], stacked=True, colormap="Set2")
        axes[1, 1].set_ylabel("Number of Patients", fontsize=11, fontweight="bold")
        axes[1, 1].set_xlabel("Chief Complaint", fontsize=11, fontweight="bold")
        axes[1, 1].set_title("Patient Disposition by Complaint", fontsize=12, fontweight="bold")
        axes[1, 1].legend(title="Disposition", loc="upper right")
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        if save:
            plt.savefig(f"{self.fig_dir}/03_complaint_analysis.png", dpi=300, bbox_inches='tight')
            print("✓ Saved: 03_complaint_analysis.png")
        plt.show()

    def plot_vital_signs_correlation(self, save=True):
        """Analyze vital signs correlation with LOS."""
        # Select vital columns for correlation
        vital_cols = [
            "systolic_bp", "diastolic_bp", "heart_rate",
            "respiratory_rate", "spo2", "temperature_c", "los_hours"
        ]

        # Create correlation matrix
        corr_data = self.df[vital_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title("Vital Signs Correlation with Length of Stay", fontsize=12, fontweight="bold", pad=20)

        plt.tight_layout()
        if save:
            plt.savefig(f"{self.fig_dir}/04_vital_correlation.png", dpi=300, bbox_inches='tight')
            print("✓ Saved: 04_vital_correlation.png")
        plt.show()

    def plot_wait_time_analysis(self, save=True):
        """Analyze wait times and their relationship with LOS."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Wait time distribution
        axes[0, 0].hist(self.df["wait_time_before_physician_hours"], bins=50, color="mediumseagreen", alpha=0.7, edgecolor="black")
        axes[0, 0].set_xlabel("Wait Time (hours)", fontsize=11, fontweight="bold")
        axes[0, 0].set_ylabel("Frequency", fontsize=11, fontweight="bold")
        axes[0, 0].set_title("Distribution of Wait Times Before Physician", fontsize=12, fontweight="bold")
        axes[0, 0].axvline(self.df["wait_time_before_physician_hours"].mean(), color="red", linestyle="--", linewidth=2)

        # Wait time vs LOS scatter
        axes[0, 1].scatter(self.df["wait_time_before_physician_hours"], self.df["los_hours"], 
                          alpha=0.3, s=20, color="purple")
        axes[0, 1].set_xlabel("Wait Time Before Physician (hours)", fontsize=11, fontweight="bold")
        axes[0, 1].set_ylabel("Length of Stay (hours)", fontsize=11, fontweight="bold")
        axes[0, 1].set_title("Wait Time vs LOS", fontsize=12, fontweight="bold")

        # Calculate correlation
        corr = self.df["wait_time_before_physician_hours"].corr(self.df["los_hours"])
        axes[0, 1].text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=axes[0, 1].transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Wait time by triage
        self.df.boxplot(column="wait_time_before_physician_hours", by="triage_category", ax=axes[1, 0])
        axes[1, 0].set_xlabel("Triage Category", fontsize=11, fontweight="bold")
        axes[1, 0].set_ylabel("Wait Time (hours)", fontsize=11, fontweight="bold")
        axes[1, 0].set_title("Wait Time by Triage Severity", fontsize=12, fontweight="bold")
        plt.suptitle("")

        # Wait time by disposition
        self.df.boxplot(column="wait_time_before_physician_hours", by="disposition", ax=axes[1, 1])
        axes[1, 1].set_xlabel("Disposition", fontsize=11, fontweight="bold")
        axes[1, 1].set_ylabel("Wait Time (hours)", fontsize=11, fontweight="bold")
        axes[1, 1].set_title("Wait Time by Patient Disposition", fontsize=12, fontweight="bold")
        plt.suptitle("")

        plt.tight_layout()
        if save:
            plt.savefig(f"{self.fig_dir}/05_wait_time_analysis.png", dpi=300, bbox_inches='tight')
            print("✓ Saved: 05_wait_time_analysis.png")
        plt.show()

    def plot_regional_analysis(self, save=True):
        """Analyze patient flow by region."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Patients by region
        region_counts = self.df["region"].value_counts()
        axes[0].barh(region_counts.index, region_counts.values, color="teal", alpha=0.7, edgecolor="black")
        axes[0].set_xlabel("Number of Patients", fontsize=11, fontweight="bold")
        axes[0].set_title("ED Patient Volume by Ethiopian Region", fontsize=12, fontweight="bold")

        # LOS by region
        los_by_region = self.df.groupby("region")["los_hours"].mean().sort_values()
        axes[1].barh(los_by_region.index, los_by_region.values, color="lightseagreen", alpha=0.7, edgecolor="black")
        axes[1].set_xlabel("Mean LOS (hours)", fontsize=11, fontweight="bold")
        axes[1].set_title("Average ED Stay by Patient Region of Origin", fontsize=12, fontweight="bold")

        plt.tight_layout()
        if save:
            plt.savefig(f"{self.fig_dir}/06_regional_analysis.png", dpi=300, bbox_inches='tight')
            print("✓ Saved: 06_regional_analysis.png")
        plt.show()

    def plot_demographics(self, save=True):
        """Analyze demographic patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Age distribution
        axes[0, 0].hist(self.df["age"], bins=50, color="skyblue", alpha=0.7, edgecolor="black")
        axes[0, 0].set_xlabel("Age (years)", fontsize=11, fontweight="bold")
        axes[0, 0].set_ylabel("Frequency", fontsize=11, fontweight="bold")
        axes[0, 0].set_title("Age Distribution of ED Patients", fontsize=12, fontweight="bold")

        # Gender distribution
        gender_counts = self.df["gender"].value_counts()
        axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct="%1.1f%%", colors=["lightcoral", "lightskyblue"])
        axes[0, 1].set_title("Gender Distribution", fontsize=12, fontweight="bold")

        # Transport mode
        transport_counts = self.df["transport_mode"].value_counts()
        axes[1, 0].barh(transport_counts.index, transport_counts.values, color="lightgreen", alpha=0.7, edgecolor="black")
        axes[1, 0].set_xlabel("Number of Patients", fontsize=11, fontweight="bold")
        axes[1, 0].set_title("Patient Transport Mode", fontsize=12, fontweight="bold")

        # LOS by age group
        age_groups = pd.cut(self.df["age"], bins=[0, 5, 18, 35, 60, 100], labels=["Infant", "Child", "Adult", "Senior", "Elderly"])
        age_group_los = self.df.groupby(age_groups)["los_hours"].agg(["mean", "std"])
        axes[1, 1].bar(range(len(age_group_los)), age_group_los["mean"], 
                       yerr=age_group_los["std"], capsize=5, color="orange", alpha=0.7, edgecolor="black")
        axes[1, 1].set_xticks(range(len(age_group_los)))
        axes[1, 1].set_xticklabels(age_group_los.index)
        axes[1, 1].set_ylabel("Mean LOS ± SD (hours)", fontsize=11, fontweight="bold")
        axes[1, 1].set_title("LOS by Age Group", fontsize=12, fontweight="bold")

        plt.tight_layout()
        if save:
            plt.savefig(f"{self.fig_dir}/07_demographics.png", dpi=300, bbox_inches='tight')
            print("✓ Saved: 07_demographics.png")
        plt.show()

    def print_summary_statistics(self):
        """Print key summary statistics."""
        print("\n" + "="*70)
        print("ETHIOPIAN ED DATA - SUMMARY STATISTICS")
        print("="*70)

        print(f"\nDataset Size: {len(self.df):,} patients")

        print(f"\n--- LENGTH OF STAY (Primary Target) ---")
        print(f"  Mean: {self.df['los_hours'].mean():.2f} hours")
        print(f"  Median: {self.df['los_hours'].median():.2f} hours")
        print(f"  Std Dev: {self.df['los_hours'].std():.2f} hours")
        print(f"  Min: {self.df['los_hours'].min():.2f} hours")
        print(f"  Max: {self.df['los_hours'].max():.2f} hours")
        print(f"  IQR: {self.df['los_hours'].quantile(0.75) - self.df['los_hours'].quantile(0.25):.2f} hours")

        print(f"\n--- WAIT TIME BEFORE PHYSICIAN ---")
        print(f"  Mean: {self.df['wait_time_before_physician_hours'].mean():.2f} hours")
        print(f"  Median: {self.df['wait_time_before_physician_hours'].median():.2f} hours")

        print(f"\n--- TRIAGE DISTRIBUTION ---")
        triage_dist = self.df["triage_category"].value_counts().sort_index()
        triage_labels = {1: "Resuscitation", 2: "Emergent", 3: "Urgent", 4: "Semi-urgent", 5: "Non-urgent"}
        for triage_id, count in triage_dist.items():
            print(f"  {triage_labels[triage_id]:20s}: {count:5d} ({count/len(self.df)*100:5.1f}%)")

        print(f"\n--- DISPOSITION ---")
        disposition_dist = self.df["disposition"].value_counts()
        for disp, count in disposition_dist.items():
            print(f"  {disp:20s}: {count:5d} ({count/len(self.df)*100:5.1f}%)")

        print(f"\n--- TOP CHIEF COMPLAINTS ---")
        complaint_dist = self.df["chief_complaint"].value_counts().head(10)
        for complaint, count in complaint_dist.items():
            print(f"  {complaint:30s}: {count:5d} ({count/len(self.df)*100:5.1f}%)")

        print(f"\n--- MISSING DATA ---")
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        if len(missing_pct) > 0:
            for col, pct in missing_pct.items():
                print(f"  {col:30s}: {pct:5.1f}%")
        else:
            print("  None")

        print("\n" + "="*70)

    def run_all_analyses(self):
        """Run all EDA analyses."""
        print("\nStarting Exploratory Data Analysis...")
        self.print_summary_statistics()

        print("\nGenerating visualizations...")
        self.plot_los_distribution()
        self.plot_peak_hours_analysis()
        self.plot_complaint_analysis()
        self.plot_vital_signs_correlation()
        self.plot_wait_time_analysis()
        self.plot_regional_analysis()
        self.plot_demographics()

        print("\n✓ All analyses completed!")


def main():
    """Run EDA."""
    df = pd.read_csv("/home/betheln/projects/EthioHealth-AI/data/synthetic_ed_data.csv")
    analyzer = EthiopianEDAnalyzer(df)
    analyzer.run_all_analyses()


if __name__ == "__main__":
    main()
