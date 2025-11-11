import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def ladda_data():
    "Ladda data från CSV-filen eller skapa testdata"
    try:
       
        df = pd.read_csv('health_study_dataset.csv')
        print("Data laddad från CSV-fil")
        return df
    except FileNotFoundError:
        
        
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
    "Beräkna grundläggande statistik"
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
    "Skapa 4 olika grafer"
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
    "Simulera sjukdomsförekomst - FIXED VERSION"
    andel_sjuk_verklig = data['disease'].mean()
    
    
    np.random.seed(42)
    # Simulera 1000 personer med samma sannolikhet
    simulerade_sjukdomar = np.random.choice([0, 1], 
                                          size=1000, 
                                          p=[1-andel_sjuk_verklig, andel_sjuk_verklig])
    
    andel_sjuk_simulerad = simulerade_sjukdomar.mean()
    
    print(f"Verklig andel sjuka: {andel_sjuk_verklig:.3f}")
    print(f"Simulerad andel sjuka: {andel_sjuk_simulerad:.3f}")
    print(f"Skillnad: {abs(andel_sjuk_verklig - andel_sjuk_simulerad):.3f}")
    
    return andel_sjuk_verklig, andel_sjuk_simulerad, simulerade_sjukdomar


def konfidensintervall(data):
    "Beräkna konfidensintervall för blodtryck"
    bp = data['systolic_bp']
    n = len(bp)
    medel = bp.mean()
    std_err = stats.sem(bp)
    ci = stats.t.interval(0.95, df=n-1, loc=medel, scale=std_err)
    return medel, ci[0], ci[1]

def hypotesprov(data):
    """Testa hypotesen att rökare har högre blodtryck - IMPROVED VERSION"""
    rokare_bp = data[data['smoker'] == 'Yes']['systolic_bp']
    icke_rokare_bp = data[data['smoker'] == 'No']['systolic_bp']
    
    t_stat, p_value = stats.ttest_ind(rokare_bp, icke_rokare_bp, equal_var=False)
    
    
    p_value_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value/2
    
    print(f"Rökare medel BP: {rokare_bp.mean():.2f}")
    print(f"Icke-rökare medel BP: {icke_rokare_bp.mean():.2f}")
    print(f"Skillnad: {rokare_bp.mean() - icke_rokare_bp.mean():.2f}")
    print(f"P-värde (enriktad): {p_value_one_sided:.4f}")
    
    if p_value_one_sided < 0.05:
        print(" Signifikant skillnad - rökare har högre blodtryck")
    else:
        print(" Ingen signifikant skillnad")
    
    return rokare_bp.mean(), icke_rokare_bp.mean(), p_value_one_sided, t_stat


def skapa_konfidensintervall_graf(data):
    "Skapa graf för konfidensintervall"
    
    bp = data['systolic_bp']
    n = len(bp)
    medel = bp.mean()
    std_err = stats.sem(bp)
    ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=medel, scale=std_err)
    
    plt.figure(figsize=(10, 6))
    plt.hist(data['systolic_bp'], bins=20, alpha=0.7, color='lightblue', edgecolor='black', density=True)
    plt.axvline(medel, color='red', linestyle='--', linewidth=2, label=f'Medelvärde: {medel:.1f}')
    plt.axvline(ci_low, color='orange', linestyle=':', linewidth=2, label=f'95% CI nedre: {ci_low:.1f}')
    plt.axvline(ci_high, color='orange', linestyle=':', linewidth=2, label=f'95% CI övre: {ci_high:.1f}')
    plt.xlabel('Systoliskt Blodtryck (mmHg)')
    plt.ylabel('Densitet')
    plt.title('Konfidensintervall för Systoliskt Blodtryck')
    plt.legend()
    plt.show()

def skapa_hypotesgraf(data):
    "Skapa graf för hypotesprövning"
    rokare_bp = data[data['smoker'] == 'Yes']['systolic_bp']
    icke_rokare_bp = data[data['smoker'] == 'No']['systolic_bp']
    
    plt.figure(figsize=(10, 6))
    plt.boxplot([rokare_bp, icke_rokare_bp], labels=['Rökare', 'Icke-rökare'])
    plt.ylabel('Systoliskt Blodtryck (mmHg)')
    plt.title('Jämförelse av Blodtryck mellan Rökare och Icke-rökare')
    plt.grid(True, alpha=0.3)
    plt.show()