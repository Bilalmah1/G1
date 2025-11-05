import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ladda_data():
    np.random.seed(42)
    n = 800
    data = pd.DataFrame({
        'age': np.random.normal(50, 15, n).astype(int),
        'sex': np.random.choice(['M', 'F'], n),
        'height': np.random.normal(170, 10, n).astype(int),
        'weight': np.random.normal(75, 15, n).astype(int),
        'systolic_bp': np.random.normal(140, 20, n).astype(int),
        'cholesterol': np.random.normal(5, 1, n),
        'smoker': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
        'disease': np.random.choice([0, 1], n, p=[0.85, 0.15])
    })
    return data

def grundstatistik(data):
    kolumner = ['age', 'weight', 'height', 'systolic_bp', 'cholesterol']
    stats = pd.DataFrame({
        col: [data[col].mean(), data[col].median(), data[col].min(), data[col].max()] 
        for col in kolumner
    }, index=['Medel', 'Median', 'Min', 'Max'])
    return stats.round(2)

def skapa_grafer(data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Stapel: Antal personer per åldersgrupp
    data['åldersgrupp'] = pd.cut(data['age'], bins=[0, 30, 50, 70, 100], labels=['<30', '30-50', '50-70', '70+'])
    antal_per_ålder = data['åldersgrupp'].value_counts().sort_index()
    axes[0].bar(antal_per_ålder.index, antal_per_ålder.values, color='lightgreen', alpha=0.7)
    axes[0].set_title('Antal personer per åldersgrupp')
    axes[0].set_ylabel('Antal personer')
    
    # 2. Stapel: Andel rökare
    rokare = data['smoker'].value_counts()
    axes[1].bar(rokare.index, rokare.values, color=['lightgreen', 'lightcoral'], alpha=0.7)
    axes[1].set_title('Fördelning rökare')
    axes[1].set_ylabel('Antal personer')
    
    # 3. Stapel: Andel sjuka
    sjuka = data['disease'].value_counts()
    axes[2].bar(['Friska', 'Sjuka'], sjuka.values, color=['lightblue', 'salmon'], alpha=0.7)
    axes[2].set_title('Sjukdomsfördelning')
    axes[2].set_ylabel('Antal personer')
    
    plt.tight_layout()
    return fig

def simulera_sjukdom(data):
    andel_sjuk = data['disease'].mean()
    simulerad = np.random.choice([0, 1], size=1000, p=[1-andel_sjuk, andel_sjuk])
    return andel_sjuk, simulerad.mean(), simulerad

def konfidensintervall(data):
    bp = data['systolic_bp']
    medel = bp.mean()
    std = bp.std()
    n = len(bp)
    felmarginal = 1.96 * (std / np.sqrt(n))
    return medel, medel - felmarginal, medel + felmarginal

def hypotesprov(data):
    rokare_bp = data[data['smoker'] == 'Yes']['systolic_bp']
    icke_rokare_bp = data[data['smoker'] == 'No']['systolic_bp']
    return rokare_bp.mean(), icke_rokare_bp.mean()