from catboost import CatBoostRegressor
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import pandas as pd
import math
import datetime as dt
from pathlib import Path

import data


# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Ekovin forecast app',
    page_icon=':sun_behind_rain_cloud:', # This is an emoji shortcode. Could be a URL too.
)

utc_now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
last_gfs_forecast_dt = utc_now - dt.timedelta(hours=3.5)  # delay (TODO: check what actually is the latest forecast?)
last_gfs_forecast_dt = pd.to_datetime(last_gfs_forecast_dt).floor("6h").to_pydatetime()  # starts at 0, 6, 12, 18


@st.cache_data(show_spinner="Downloading latest GFS data", ttl=60*60)
def download_latest_gfs(lead_time):
    data.gfs.download_dataframe(
        current_date=last_gfs_forecast_dt,
        lead_time_hours=lead_time,
        source="nomads",
        lat_min=45, lat_max=55, lon_min=8, lon_max=22,
    ).to_csv("latest_gfs.csv")

    return data.preprocess.load_gfs("latest_gfs.csv")


@st.cache_data(show_spinner="Downloading latest station data", ttl=15*60)
def download_latest_ekovin(station_id, station_latitude, station_longitude):
    return data.ekovin.download(
        probe_id=station_id,
        latitude=station_latitude,
        longitude=station_longitude,
        dt_start=last_gfs_forecast_dt - dt.timedelta(hours=6),
        dt_end=last_gfs_forecast_dt + dt.timedelta(hours=6),
    ).reset_index()


station_id = 11359321
station_latitude = 48.810326
station_longitude = 16.652575
station = download_latest_ekovin(station_id, station_latitude, station_longitude)

lead_times = list(range(6, 49, 6))
lead_time_gfs = {t: download_latest_gfs(t) for t in lead_times}

def merge(areas, station):
    mapping = data.preprocess.get_mapping(areas, station)
    area_id = mapping.iloc[0].area_index
    area = areas[areas.area_index == area_id]
    return data.preprocess.merge(station, area)
lead_time_X = {t: merge(lead_time_gfs[t], station) for t in lead_times}


def predict(variable, lead_time):
    model = CatBoostRegressor().load_model(f"models/{station_id}/{lead_time}/{variable.replace('/', '')}.cbm")
    X = lead_time_X[lead_time]
    pred = model.predict(X[model.feature_names_])
    return pred[0]

target_variables = station.drop(columns=['latitude', 'longitude', 'station_index', 'datetime']).columns

with st.spinner("Predicting"):
    df = pd.DataFrame(
        {v: [predict(v, t) for t in lead_times] for v in target_variables},
        index=[lead_time_X[t].index[0] + dt.timedelta(hours=t) for t in lead_times],
    )

st.dataframe(df.style.format("{:.1f}"))

variable_name = st.selectbox("Variable", list(target_variables))

fig, ax = plt.subplots(figsize=(10, 5))

station.set_index("datetime")[variable_name].plot(ax=ax, style=".-", label="station data")
df[variable_name].plot(ax=ax, style=".--", label="forecast")

ymin, ymax = ax.get_ylim()
if -0.1 < ymin < 0 < ymax < 0.1:
    ax.set_ylim(0, 1)
ax.grid(which='minor', alpha=0.5)
ax.grid(which='major', alpha=1.0)
ax.legend()
st.pyplot(fig)
