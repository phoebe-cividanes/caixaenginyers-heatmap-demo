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

    def _compute_next(self, series: pd.Series):
        """Compute next-step prediction using SMA or EMA."""
        if self.mode == "SMA":
            vals = series.values[-self.n_elements:]
            return np.mean(vals)
        else:
            return series.ewm(span=self.n_elements, adjust=False).mean().iloc[-1]

    def predict(self, target_column: str):
        target_df = self.df[self.df["Municipios"].str.contains(target_column, case=False, na=False)].copy()
        if target_df.empty:
            raise ValueError(f"No match found for municipality '{target_column}'. Available options include: {self.cities[:5]} ...")
        target_df = target_df.sort_values("Periodo")
        target_df["Total"] = target_df["Total"].astype(str).str.replace(",", ".").astype(float)
        history = target_df.copy()

        for _ in range(self.n_years):
            new_value = self._compute_next(target_df["Total"])
            new_period = target_df["Periodo"].max() + 1
            target_df.loc[len(target_df)] = [target_df["Municipios"].iloc[0], new_period, new_value]

        return history, target_df

    def backtest(self, target_column: str):
        """Leave-one-out time-series backtesting."""
        target_df = self.df[self.df["Municipios"].str.contains(target_column, case=False, na=False)].copy()
        if target_df.empty:
            raise ValueError(f"No match found for municipality '{target_column}'.")
        target_df = target_df.sort_values("Periodo")
        target_df["Total"] = target_df["Total"].astype(str).str.replace(",", ".").astype(float)

        predictions, actuals, periods = [], [], []

        for i in range(self.n_elements, len(target_df)):
            train = target_df["Total"].iloc[:i]
            actual = target_df["Total"].iloc[i]
            pred = self._compute_next(train)
            predictions.append(pred)
            actuals.append(actual)
            periods.append(target_df["Periodo"].iloc[i])

        errors = np.array(predictions) - np.array(actuals)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        return periods, predictions, actuals, mae, rmse

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
        plt.savefig(f"prediction_{self.mode}.jpg")
        plt.show()


    def plot_backtest(self, target_column: str):
        periods, preds, acts, mae, rmse = self.backtest(target_column)

        plt.figure(figsize=(8, 5))
        plt.plot(periods, acts, marker="o", label="Actual", color="blue")
        plt.plot(periods, preds, marker="x", label=f"Predicted ({self.mode})", color="orange")
        plt.fill_between(periods, np.array(preds) - np.abs(np.array(preds) - np.array(acts)),
                         np.array(preds) + np.abs(np.array(preds) - np.array(acts)),
                         color="orange", alpha=0.1)

        plt.title(f"Backtest ({self.mode}) for {target_column}\nMAE={mae:.2f}, RMSE={rmse:.2f}")
        plt.xlabel("Periodo")
        plt.ylabel("Total")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"backtest_{self.mode}.jpg")
        plt.show()



if __name__ == "__main__":
    df = pd.read_csv("../data/popu_growth_population.csv", sep="\t")

    sma = SimpleMovingAverage(df, n_elements=4, n_years=4, mode="SMA")

    target_city = "Terrassa"
    history, predicted = sma.predict(target_city)
    sma.plot_prediction(history, predicted, target_city)

    sma.plot_backtest(target_city)

