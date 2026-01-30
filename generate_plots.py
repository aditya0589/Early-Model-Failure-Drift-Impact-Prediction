import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # Create output directory
    output_dir = "reports/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        df = pd.read_csv("data/processed/failure_training_data.csv")
    except FileNotFoundError:
        print("Data file not found!")
        return

    # Set style
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Drift vs Performance Drop
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x='drift_risk_score', 
        y='performance_drop', 
        hue='failure_label',
        palette={0: 'blue', 1: 'red'},
        alpha=0.6
    )
    plt.title("Drift Score vs Performance Drop")
    plt.axvline(x=0.1, color='green', linestyle='--', label='Drift Threshold (0.1)')
    plt.axhline(y=0.02, color='orange', linestyle='--', label='Perf Drop Threshold (0.02)')
    plt.legend()
    plt.savefig(f"{output_dir}/drift_vs_performance.png")
    print(f"Saved {output_dir}/drift_vs_performance.png")
    
    # Plot 2: Failure Label Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df['failure_label'])
    plt.title("Distribution of Failure Labels")
    plt.xlabel("Failure Label (0=Normal, 1=Failure)")
    plt.ylabel("Count")
    plt.savefig(f"{output_dir}/failure_distribution.png")
    print(f"Saved {output_dir}/failure_distribution.png")
    
    # Plot 3: Drift Type Impact
    if 'drift_type' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='drift_type', y='performance_drop')
        plt.title("Impact of Drift Type on Performance Drop")
        plt.xticks(rotation=45)
        plt.savefig(f"{output_dir}/drift_type_impact.png")
        print(f"Saved {output_dir}/drift_type_impact.png")

if __name__ == "__main__":
    main()
