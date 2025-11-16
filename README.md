# TimeSeriesForecasting

The project demonstrates baseline models, AutoARIMA, exogenous features, cross-validation, prediction intervals, and evaluation metrics.

**Repository structure**

- `streamlit_app.py`: A Streamlit app (if present) to visualize forecasts and interact with the pipeline.
- `forecast_stats.ipynb`: Notebook with examples and walkthroughs (primary analysis notebook).
- `data/`: Directory containing datasets used in the notebooks.
  - `daily_sales_french_bakery.csv`: Example daily sales dataset used across the notebooks.
- `requirements.txt`: Python dependency list.

**Quick start**

1. Create a Python environment (recommended using conda):

   ```powershell
   pip install -r requirements.txt
   ```

2. Open the main notebook with Jupyter or VS Code:

   ```powershell
   jupyter lab forecast_stats.ipynb
   ```

3. Run the Streamlit app (if you want the web UI):

   ```powershell
   streamlit run streamlit_app.py
   ```



**Data**

- The dataset `data/daily_sales_french_bakery.csv` 
- Link : https://www.kaggle.com/datasets/matthieugimbert/french-bakery-daily-sales

