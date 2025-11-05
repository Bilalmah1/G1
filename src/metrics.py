import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def ladda_data():
    """Ladda data från CSV-filen"""
    df = pd.read_csv('health_study_dataset.csv')
    return df

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
    """Skapa 3 olika grafer enligt uppgiftskrav"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. HISTOGRAM över blodtryck (krav)
    axes[0,0].hist(data['systolic_bp'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Histogram över Systoliskt Blodtryck')
    axes[0,0].set_xlabel('Blodtryck (mmHg)')
    axes[0,0].set_ylabel('Antal Personer')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. BOXPLOT över vikt per kön (krav)
    data_male = data[data['sex'] == 'M']['weight']
    data_female = data[data['sex'] == 'F']['weight']
    axes[0,1].boxplot([data_male, data_female], labels=['Män', 'Kvinnor'])
    axes[0,1].set_title('Boxplot: Viktfördelning per Kön')
    axes[0,1].set_ylabel('Vikt (kg)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. STAPELDIAGRAM över andelen rökare (krav)
    smoker_counts = data['smoker'].value_counts()
    axes[1,0].bar(smoker_counts.index, smoker_counts.values, 
                 color=['lightgreen', 'lightcoral'], alpha=0.7)
    axes[1,0].set_title('Fördelning av Rökare')
    axes[1,0].set_xlabel('Rökare')
    axes[1,0].set_ylabel('Antal Personer')
    
    # 4. EXTRA: Sjukdomsfördelning
    disease_counts = data['disease'].value_counts()
    axes[1,1].bar(['Friska', 'Sjuka'], disease_counts.values, 
                 color=['lightblue', 'salmon'], alpha=0.7)
    axes[1,1].set_title('Sjukdomsfördelning')
    axes[1,1].set_ylabel('Antal Personer')
    
    plt.tight_layout()
    return fig

def simulera_sjukdom(data):
    """Simulera sjukdomsförekomst"""
    # Beräkna verklig andel
    andel_sjuk_verklig = data['disease'].mean()
    
    # Simulera 1000 personer med samma sannolikhet
    np.random.seed(42)
    simulerad_data = np.random.binomial(n=1, p=andel_sjuk_verklig, size=1000)
    andel_sjuk_simulerad = simulerad_data.mean()
    
    return andel_sjuk_verklig, andel_sjuk_simulerad, simulerad_data

def konfidensintervall(data):
    """Beräkna konfidensintervall för blodtryck"""
    bp = data['systolic_bp']
    n = len(bp)
    medel = bp.mean()
    std_err = stats.sem(bp)  # Standard error
    
    # 95% konfidensintervall med t-fördelning
    ci = stats.t.interval(0.95, df=n-1, loc=medel, scale=std_err)
    
    return medel, ci[0], ci[1]

def hypotesprov(data):
    """Testa hypotesen att rökare har högre blodtryck"""
    # Separera data
    rokare_bp = data[data['smoker'] == 'Yes']['systolic_bp']
    icke_rokare_bp = data[data['smoker'] == 'No']['systolic_bp']
    
    # Beräkna medelvärden
    medel_rokare = rokare_bp.mean()
    medel_icke = icke_rokare_bp.mean()
    
    # Statistisk test (t-test)
    t_stat, p_value = stats.ttest_ind(rokare_bp, icke_rokare_bp, equal_var=False)
    
    return medel_rokare, medel_icke, p_value, t_stat