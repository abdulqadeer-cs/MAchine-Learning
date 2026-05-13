import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    housing = fetch_california_housing(as_frame=True)
    x = housing.data
    y = housing.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print("=== House Price Predictor ===")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    sample_features = x_test.head(5)
    sample_preds = model.predict(sample_features)
    sample_df = pd.DataFrame(
        {
            "ActualPrice": y_test.head(5).values,
            "PredictedPrice": sample_preds,
        }
    )
    print("\nSample Predictions (first 5):")
    print(sample_df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, predictions, alpha=0.35)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.tight_layout()
    plt.savefig("prediction_scatter.png", dpi=120)
    print("\nSaved plot: prediction_scatter.png")


if __name__ == "__main__":
    main()
