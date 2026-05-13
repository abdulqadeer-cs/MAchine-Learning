# House Price Predictor

Beginner regression project using:
- `scikit-learn`
- `LinearRegression`
- California housing dataset

## Run

```bash
python train.py
```

## Run Streamlit App

```bash
streamlit run app.py
```

The script:
- loads a real housing dataset
- trains a linear regression model
- prints MAE, MSE, and R2
- saves `prediction_scatter.png` (actual vs predicted)
