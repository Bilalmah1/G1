import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    np.random.seed(42)
    n = 800
    data = pd.DataFrame({
        'age': np.random.normal(50, 15, n).astype(int),
        'sex': np.random.choice(['M', 'F'], n),
        'height': np.random.normal(170, 10, n),
        'weight': np.random.normal(75, 15, n),
        'systolic_bp': np.random.normal(140, 20, n),
        'cholesterol': np.random.normal(5, 1, n),
        'smoker': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
        'disease': np.random.choice([0, 1], n, p=[0.85, 0.15])
    })
    return data

def basic_stats(data):
    kolumner = ['age', 'weight', 'height', 'systolic_bp', 'cholesterol']
    stats = pd.DataFrame({
        col: [data[col].mean(), data[col].median(), data[col].min(), data[col].max()] 
        for col in kolumner
    }, index=['Medel', 'Median', 'Min', 'Max'])
    return stats.round(2)

def create_plots(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram
    axes[0].hist(data['systolic_bp'], bins=15, color='lightblue')
    axes[0].set_title('Blodtryck')
    axes[0].set_xlabel('mmHg')
    
    # Boxplot
    man = data[data['sex'] == 'M']['weight']
    kvinna = data[data['sex'] == 'F']['weight']
    axes[1].boxplot([man, kvinna], labels=['Män', 'Kvinnor'])
    axes[1].set_title('Vikt per kön')
    axes[1].set_ylabel('kg')
    
    # Bar chart
    rokare = data['smoker'].value_counts()
    axes[2].bar(rokare.index, rokare.values, color=['lightgreen', 'lightcoral'])
    axes[2].set_title('Rökare')
    
    plt.tight_layout()
    return fig

def simulate_disease(data):
    andel_sjuk = data['disease'].mean()
    simulerad = np.random.choice([0, 1], size=1000, p=[1-andel_sjuk, andel_sjuk])
    return andel_sjuk, simulerad.mean(), simulerad

def confidence_interval(data):
    bp = data['systolic_bp']
    medel = bp.mean()
    std = bp.std()
    n = len(bp)
    felmarginal = 1.96 * (std / np.sqrt(n))
    return medel, medel - felmarginal, medel + felmarginal

def hypothesis_test(data):
    rokare_bp = data[data['smoker'] == 'Yes']['systolic_bp']
    icke_rokare_bp = data[data['smoker'] == 'No']['systolic_bp']
    return rokare_bp.mean(), icke_rokare_bp.mean()