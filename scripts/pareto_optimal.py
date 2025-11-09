import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("merged_es_dropna.csv")

#df["poblacion_65_mas"] = df["poblacion_65_mas"] / df["poblacion_65_mas"].max()
#df["renta_bruta_media"] = df["renta_bruta_media"] / df["renta_bruta_media"].max()

df["Elderly_ROI"] = df["poblacion_65_mas"] * df["renta_bruta_media"]
Q1 = df["poblacion_65_mas"].quantile(0.25)
Q3 = df["poblacion_65_mas"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df["poblacion_65_mas"] >= lower_bound) & (df["poblacion_65_mas"] <= upper_bound)]
df = df.sort_values("Elderly_ROI", ascending=False)


df_sorted = df.sort_values("poblacion_65_mas")
pareto = [df_sorted.iloc[0]]
for _, row in df_sorted.iterrows():
    if row["renta_bruta_media"] >= pareto[-1]["renta_bruta_media"]:
        pareto.append(row)
pareto_df = pd.DataFrame(pareto)


plt.figure(figsize=(10,7))

plt.scatter(df["poblacion_65_mas"], df["renta_bruta_media"],
            color="lightgray", label="Regions", s=50, alpha=0.6)

plt.scatter(pareto_df["poblacion_65_mas"], pareto_df["renta_bruta_media"],
            color="red", label="Pareto Frontier", s=100, zorder=5)

plt.plot(pareto_df["poblacion_65_mas"], pareto_df["renta_bruta_media"],
         color="red", linestyle="--", alpha=0.7)

for _, row in pareto_df.iterrows():
    plt.text(row["poblacion_65_mas"]+0.3, row["renta_bruta_media"]+200,
             row["municipio"],
             fontsize=9, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

plt.title("Pareto Frontier: Elderly Density vs Household Income", fontsize=14)
plt.xlabel("Elderly Density (>65)", fontsize=12)
plt.ylabel("Household Income (€)", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("Pareto-optimal municipios:") 
print(pareto_df["municipio"].tolist())


