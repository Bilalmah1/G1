import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

class HealthAnalyzer:
    "Klass för att utföra fördjupad analys på hälsodata."

    def __init__(self, data: pd.DataFrame):
        
        self.data = data

    def regression_bp(self):
        "Utför enkel linjär regression för att förutsäga blodtryck från ålder och vikt."
        X = self.data[["age", "weight"]]
        y = self.data["systolic_bp"]

        model = LinearRegression()
        model.fit(X, y)
        self.model = model

        print("Regression Genomförd!")
        print(f"Intercept: {model.intercept_:.2f}")
        print(f"Koefficienter: {dict(zip(X.columns, model.coef_.round(2)))}")
        return model

    def plot_regression(self):
        "Ritar ett spridningsdiagram mellan ålder och blodtryck."
        plt.scatter(self.data["age"], self.data["systolic_bp"], alpha=0.5, label="Observationer")
        plt.xlabel("Ålder")
        plt.ylabel("Systoliskt blodtryck (mmHg)")
        plt.title("Samband mellan ålder och blodtryck")
        plt.legend()
        plt.show()

    def pca_analysis(self):
        "Utför PCA på numeriska variabler för att hitta mönster."
        numeric = self.data.select_dtypes(include=np.number)
        pca = PCA(n_components=2)
        components = pca.fit_transform(numeric)
        print("Förklarad varians:", np.round(pca.explained_variance_ratio_, 3))
        plt.scatter(components[:, 0], components[:, 1], alpha=0.5)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA – mönster i hälsodata")
        plt.show()

    def sjukdom_per_koen(self):
         "Visar andel sjuka per kön."
         mean_values = self.data.groupby("sex")["disease"].mean()
         mean_values.plot(kind="bar", color=["lightblue", "salmon"])
         plt.ylabel("Andel sjuka")
         plt.title("Sjukdomsförekomst per kön")
         plt.show()