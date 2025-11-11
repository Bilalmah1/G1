import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def ladda_data():
    """Ladda data från CSV-filen eller skapa testdata"""
    try:
        # Försök ladda CSV
        df = pd.read_csv('health_study_dataset.csv')
        print("Data laddad från CSV-fil")
        return df
    except FileNotFoundError:
        print("CSV-fil inte hittad. Skapar testdata...")
        # Skapa testdata som liknar originalet
        np.random.seed(42)
        n = 800
        data = pd.DataFrame({
            'id': range(1, n+1),
            'age': np.random.normal(50, 15, n).clip(18, 90).astype(int),
            'sex': np.random.choice(['M', 'F'], n, p=[0.5, 0.5]),
            'height': np.random.normal(170, 10, n).clip(140, 200).astype(int),
            'weight': np.random.normal(75, 15, n).clip(40, 150).astype(int),
            'systolic_bp': np.random.normal(140, 20, n).clip(100, 200).astype(int),
            'cholesterol': np.random.normal(5.0, 1.0, n).clip(2.0, 8.0),
            'smoker': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
            'disease': np.random.choice([0, 1], n, p=[0.85, 0.15])
        })
        return data

def grundstatistik(data):
    """Beräkna grundläggande statistik"""
    kolumner = ['age', 'weight', 'height', 'systolic_bp', 'cholesterol']
    
    stats_data = {}
    for col in kolumner:
        stats_data[col] = {
            'Medel': data[col].mean(),
            'Median': data[col].median(), 
            'Min': data[col].min(),
            'Max': data[col].max(),
            'Standardavvikelse': data[col].std()
        }
    
    stats_df = pd.DataFrame(stats_data).T
    return stats_df.round(2)

def skapa_grafer(data):
    """Skapa 4 olika grafer"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Histogram över blodtryck
    axes[0,0].hist(data['systolic_bp'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Histogram över Systoliskt Blodtryck')
    axes[0,0].set_xlabel('Blodtryck (mmHg)')
    axes[0,0].set_ylabel('Antal Personer')
    
    # 2. Boxplot över vikt per kön
    data_male = data[data['sex'] == 'M']['weight']
    data_female = data[data['sex'] == 'F']['weight']
    axes[0,1].boxplot([data_male, data_female], labels=['Män', 'Kvinnor'])
    axes[0,1].set_title('Boxplot: Vikt per Kön')
    axes[0,1].set_ylabel('Vikt (kg)')
    
    # 3. Stapeldiagram över rökare
    smoker_counts = data['smoker'].value_counts()
    axes[1,0].bar(smoker_counts.index, smoker_counts.values, color=['lightgreen', 'lightcoral'])
    axes[1,0].set_title('Fördelning av Rökare')
    axes[1,0].set_ylabel('Antal Personer')
    
    # 4. Extra: Sjukdomsfördelning
    disease_counts = data['disease'].value_counts()
    axes[1,1].bar(['Friska', 'Sjuka'], disease_counts.values, color=['lightblue', 'salmon'])
    axes[1,1].set_title('Sjukdomsfördelning')
    axes[1,1].set_ylabel('Antal Personer')
    
    plt.tight_layout()
    return fig

def simulera_sjukdom(data):
    """Simulera sjukdomsförekomst"""
    andel_sjuk_verklig = data['disease'].mean()
    np.random.seed(42)
    simulerad_data = np.random.binomial(n=1, p=andel_sjuk_verklig, size=1000)
    andel_sjuk_simulerad = simulerad_data.mean()
    return andel_sjuk_verklig, andel_sjuk_simulerad, simulerad_data

def konfidensintervall(data):
    """Beräkna konfidensintervall för blodtryck"""
    bp = data['systolic_bp']
    n = len(bp)
    medel = bp.mean()
    std_err = stats.sem(bp)
    ci = stats.t.interval(0.95, df=n-1, loc=medel, scale=std_err)
    return medel, ci[0], ci[1]

def hypotesprov(data):
    """Testa hypotesen att rökare har högre blodtryck"""
    rokare_bp = data[data['smoker'] == 'Yes']['systolic_bp']
    icke_rokare_bp = data[data['smoker'] == 'No']['systolic_bp']
    t_stat, p_value = stats.ttest_ind(rokare_bp, icke_rokare_bp, equal_var=False)
    return rokare_bp.mean(), icke_rokare_bp.mean(), p_value, t_stat