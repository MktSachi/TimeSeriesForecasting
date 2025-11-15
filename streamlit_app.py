import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta


st.set_page_config(page_title="Bakery Sales Forecast", layout="wide")

st.title("Bakery Sales Forecast Dashboard")


@st.cache_data
def load_data(path="data/daily_sales_french_bakery.csv"):
    df = pd.read_csv(path, parse_dates=["ds"])
    # keep only series with at least 28 points (same filter as notebook)
    df = df.groupby('unique_id').filter(lambda x: len(x) >= 28).reset_index(drop=True)
    return df


df = load_data()

products = sorted(df['unique_id'].unique())

with st.sidebar:
    st.header("Controls")
    selected = st.multiselect("Select products to show", products, default=[products[0]])
    horizon = st.slider("Forecast horizon (days)", 1, 28, 7)
    show_metrics = st.checkbox("Show simple evaluation (MAE) using last horizon", value=True)

if not selected:
    st.warning("Select at least one product from the sidebar.")
    st.stop()


def naive_forecast(series, h):
    # naive: repeat last observed value
    last = series.iloc[-1]
    return np.repeat(last, h)


def seasonal_naive_forecast(series, h, season=7):
    # seasonal naive: use last 'season' values cyclically
    last_season = series.iloc[-season:]
    reps = int(np.ceil(h / season))
    vals = np.tile(last_season.values, reps)[:h]
    return vals


def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


cols = st.columns(len(selected))

metrics_rows = []

for col, pid in zip(cols, selected):
    with col:
        st.subheader(pid)
        product_df = df[df['unique_id'] == pid].sort_values('ds').reset_index(drop=True)

        fig = px.line(product_df, x='ds', y='y', title=f"{pid} - Historical sales", labels={'y':'Sales','ds':'Date'})
        st.plotly_chart(fig, use_container_width=True)

        # prepare train/test split using last `horizon` as test (to mimic notebook evaluation)
        if len(product_df) <= horizon:
            st.info("Not enough history to compute forecast for this product.")
            continue

        train = product_df.iloc[:-horizon]
        test = product_df.iloc[-horizon:]

        naive_pred = naive_forecast(train['y'], horizon)
        snaive_pred = seasonal_naive_forecast(train['y'], horizon, season=7)

        # build a dataframe to plot forecast appended to history
        last_date = product_df['ds'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]

        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'Naive': naive_pred,
            'SeasonalNaive': snaive_pred
        })

        # plot history + forecasts
        history = product_df[['ds','y']].rename(columns={'y':'History'})
        plot_df = pd.concat([
            history.set_index('ds'),
            forecast_df.set_index('ds')
        ], axis=0).reset_index()

        fig2 = px.line(plot_df, x='ds', y=[c for c in plot_df.columns if c!='ds'], labels={'value':'Sales','ds':'Date'})
        fig2.update_layout(title=f"{pid} - Forecasts (next {horizon} days)")
        st.plotly_chart(fig2, use_container_width=True)

        if show_metrics:
            # if actuals for the horizon exist in original data beyond train, use them, else compare vs test constructed above
            mae_naive = compute_mae(test['y'].values, naive_pred[:len(test)])
            mae_snaive = compute_mae(test['y'].values, snaive_pred[:len(test)])

            st.markdown("**Evaluation on last available window**")
            st.metric(label="Naive MAE", value=f"{mae_naive:.3f}")
            st.metric(label="Seasonal Naive MAE", value=f"{mae_snaive:.3f}")

            metrics_rows.append({
                'product': pid,
                'naive_mae': mae_naive,
                'seasonal_naive_mae': mae_snaive
            })

if show_metrics and metrics_rows:
    st.header("Summary metrics")
    metrics_df = pd.DataFrame(metrics_rows)
    st.dataframe(metrics_df.set_index('product'))

    # bar chart compare models across products
    melted = metrics_df.melt(id_vars='product', var_name='model', value_name='mae')
    figm = px.bar(melted, x='product', y='mae', color='model', barmode='group', title='MAE by product and model')
    st.plotly_chart(figm, use_container_width=True)


st.sidebar.markdown("---")
st.sidebar.markdown("Run with: `streamlit run streamlit_app.py` from the project root.")
