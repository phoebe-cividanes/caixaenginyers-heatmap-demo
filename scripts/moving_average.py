import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class SimpleMovingAverage:
    def __init__(self, df: pd.DataFrame, n_elements: int = 4, n_years: int = 4, mode: str = "SMA"):
        """
        mode: "SMA" or "EMA"
        n_elements: window size
        n_years: number of forecast steps
        """
        assert mode in ["SMA", "EMA"], "Mode must be 'SMA' or 'EMA'"
        self.df = df
        self.n_elements = n_elements
        self.n_years = n_years
        self.mode = mode
        self.cities = list(df["Municipios"].unique())

    def predict(self, target_column: str):
        target_df = self.df[self.df["Municipios"] == target_column].copy()
        target_df = target_df.sort_values("Periodo")
        target_df["Total"] = target_df["Total"].astype(str).str.replace(",", ".").astype(float)
        history = target_df.copy()

        for _ in range(self.n_years):
            if self.mode == "SMA":
                vals = target_df["Total"].values[-self.n_elements:]
                new_value = sum(vals) / len(vals)
            else:  # EMA
                vals = pd.Series(target_df["Total"].values)
                new_value = vals.ewm(span=self.n_elements, adjust=False).mean().iloc[-1]

            new_period = target_df["Periodo"].max() + 1
            target_df.loc[len(target_df)] = [target_column, new_period, new_value]

        return history, target_df

    def plot_prediction(self, history: pd.DataFrame, predicted: pd.DataFrame, target_column: str):
        plt.figure(figsize=(8, 5))

        plt.plot(history["Periodo"], history["Total"], label="Historic", color="blue", marker="o")
        plt.plot(predicted["Periodo"], predicted["Total"],
                 label=f"Predicted ({self.mode})", color="orange", marker="x")

        plt.axvline(history["Periodo"].max(), color="gray", linestyle="--", alpha=0.7)
        plt.title(f"{self.mode} Forecast for {target_column}")
        plt.xlabel("Periodo")
        plt.ylabel("Total")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv("../data/popu_growth_population.csv", sep="\t")

    sma = SimpleMovingAverage(df, n_elements=8, n_years=4, mode="SMA")

    target_city = "33066 Siero" 
    history, predicted = sma.predict(target_city)
    sma.plot_prediction(history, predicted, target_city)

