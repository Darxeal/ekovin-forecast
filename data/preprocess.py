import sys
import argparse
import pandas as pd
import numpy as np
import os


def load_data(path: str):
    _, file_extension = os.path.splitext(path)
    
    if file_extension == '.csv':
        df = pd.read_csv(path)
    elif file_extension == '.parquet':
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .parquet file.")
    
    return df

def compute_dewpoints(q: np.ndarray, T: np.ndarray, p: float = 1013.25) -> np.ndarray:
    specific_humidity = q / (1 + q)
    e = (specific_humidity * p) / (0.622 + specific_humidity)
    dew_point_temp = (243.5 * np.log(e / 6.112)) / (17.67 - np.log(e / 6.112))
    return pd.Series(dew_point_temp, name='computed_dewpoints')

def load_gfs(path: str):
    df = load_data(path)
    df['datetime'] = df['datetime'].apply(lambda x: x if len(str(x)) == 19 else x + ' 00:00:00')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.drop_duplicates(subset=['datetime', 'latitude', 'longitude'])

    # these variables are not present in the realtime data for some reason
    df.drop(columns=[
        'atmosphereSingleLayer_avg_cwork',
        'highCloudLayer_avg_tcc',
        'lowCloudLayer_avg_tcc',
        'middleCloudLayer_avg_tcc',
        'surface_instant_lftx4',
    ], errors="ignore", inplace=True)

    df['pressureFromGroundLayer_instant_dewpoints'] = compute_dewpoints(df['pressureFromGroundLayer_instant_q'], 
                                                                        df['pressureFromGroundLayer_instant_t'])
    df['sigma_instant_windspeeds'] = np.sqrt(df['sigma_instant_u']**2 + df['sigma_instant_v']**2)

    # TODO: would have to download 4 most recent GFS forecasts for realtime prediction instead of just 1
    # df["surface_accum_tp_24h"] = df.groupby(["latitude", "longitude"]).surface_accum_tp.transform(lambda x: x.rolling(window=4, min_periods=0).sum())

    lat_lon_pairs = df['latitude'].astype(str) + ',' + df['longitude'].astype(str)
    df['area_index'] = pd.factorize(lat_lon_pairs)[0]

    temp_cols = [col for col in df.columns if col.endswith('t')]
    df[temp_cols] = df[temp_cols] - 273.15

    return df.sort_values(by='datetime')

def load_ekovin(path: str):
    df = load_data(path)
    df.datetime = pd.to_datetime(df.datetime)
    df = df.set_index("datetime", drop=False)
    return df

def get_mapping(areas: pd.DataFrame, stations: pd.DataFrame):
    stations_grouped = stations[['latitude', 'longitude', 'station_index']].groupby(
        ['latitude', 'longitude', 'station_index'])
    stations_grouped = stations_grouped.size().reset_index(name='Count').sort_values(
        by=['latitude', 'longitude'])
    
    areas_grouped = areas[['latitude', 'longitude', 'area_index']].groupby(
        ['latitude', 'longitude', 'area_index'])
    areas_grouped = areas_grouped.size().reset_index(name='Count').sort_values(
        by=['latitude', 'longitude'])

    mapping = pd.DataFrame({
        'station_index': stations_grouped['station_index'].astype(int),
        'station_latitude': stations_grouped['latitude'],
        'station_longitude': stations_grouped['longitude']
    })

    areas_lat, areas_lon, areas_id = [], [], []
    for _, (station_lat, station_lon) in stations_grouped[['latitude', 'longitude']].iterrows():
        for _, (area_lat, area_lon, area_id) in areas_grouped[['latitude', 'longitude', 'area_index']].iterrows():
            if abs(station_lat - area_lat) <= 0.125 and abs(station_lon - area_lon) <= 0.125:
                print(f'Station at [{station_lat}, {station_lon}] mapped to an area with ' +
                      f'center point at [{area_lat}, {area_lon}]')
                areas_lat.append(station_lat)
                areas_lon.append(station_lon)
                areas_id.append(area_id)
                break
                
    mapping['area_latitude'] = areas_lat 
    mapping['area_longitude'] = areas_lon 
    mapping['area_index'] = np.array(areas_id).astype(int)
    return mapping

def merge(station: pd.DataFrame, area: pd.DataFrame):
    station = station.rename(columns={'latitude': 'station_latitude', 'longitude': 'station_longitude'})
    area = area.drop(['longitude', 'latitude', 'area_index'], axis=1)
    area = area.rename(columns={'latitude': 'area_latitude', 'longitude': 'area_longitude'})
    merged = pd.concat([station.set_index('datetime'), area.set_index('datetime')], axis=1, join='inner').sort_index()
    merged = pd.concat([merged, 
                        pd.Series(merged.index.month, index=merged.index, name='month'),
                        pd.Series(merged.index.day, index=merged.index, name='day'),
                        pd.Series(merged.index.hour, index=merged.index, name='hour')], axis=1)
    return merged


def prepare_data(stations: pd.DataFrame, labels: list, areas: pd.DataFrame, mapping: pd.DataFrame,
                 lead_time_hours: int):
    data = {}
    for _, (station_id, area_id) in mapping[['station_index', 'area_index']].iterrows():
        station = stations[stations.station_index == station_id]
        area = areas[areas.area_index == area_id]
        merged = merge(station, area)
        
        # create labels
        merged = merged.reset_index(drop=False)
        merged['datetime_shifted'] = merged['datetime'] + pd.DateOffset(hours=lead_time_hours)
        merged_pairs = pd.merge(merged, merged, left_on='datetime_shifted', right_on='datetime', suffixes=('_original', '_shifted'))
        merged_pairs = merged_pairs[merged_pairs['datetime_original'] + pd.DateOffset(hours=lead_time_hours) == merged_pairs['datetime_shifted']]
        merged_pairs = merged_pairs.drop(['datetime_shifted_original', 'datetime_shifted_shifted', 'datetime_shifted'], axis=1)
        merged_pairs = merged_pairs.rename(columns={'datetime_original': 'datetime'})
        merged_pairs = merged_pairs.set_index('datetime')
        merged_original_cols = [col for col in merged_pairs.columns if '_original' in col]
        merged_shifted_cols = [col for col in merged_pairs.columns if '_shifted' in col]

        X = merged_pairs[merged_original_cols].rename(columns={f'{col}': col[:-9] for col in merged_original_cols})
        Y = merged_pairs.drop(merged_original_cols, axis=1).rename(columns={f'{col}': col[:-8] for col in merged_shifted_cols})
        Y = Y.rename_axis('valid_datetime')
        print(f'X shape: {X.shape}, Y shape: {Y.shape}')
        data[station_id] = (X, Y[labels])
    return data

def save_data(data: dict, output: str):
    for s_id, (X, Y) in data.items():
        if len(X) == 0:
            print(f'Station {s_id} has no data. Skipping...')
            continue
        X.to_csv(f'{output}/X_{s_id}.csv')
        Y.to_csv(f'{output}/Y_{s_id}.csv')

def main(args):
    print('Starting data preprocessing...')
    areas = load_gfs(args.areas)
    print('Areas data loaded successfully.')
    min_date, max_date = areas.datetime.min(), areas.datetime.max()
    stations = load_ekovin(args.stations, min_date, max_date)
    print('Stations data loaded successfully.')
    mapping = get_mapping(areas, stations)
    print('Mapping data created successfully.')
    data = prepare_data(stations, args.targets, areas, mapping, args.lead_time)
    print('Data prepared successfully. Saving...')
    save_data(data, args.output)
    print('Data saved successfully.')
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for Climate Forecasting Model Data Preprocessing")
    parser.add_argument("--areas", type=str, required=True, help="Path to the areas data csv. Like GFS.")
    parser.add_argument("--stations", type=str, required=True, help="Path to the stations csv.")
    parser.add_argument("--targets", nargs="+", type=str, required=True, 
                        help="List of features from the stations dataset considered as labels.")
    parser.add_argument("--lead_time", type=int, required=True, help="Lead time in hours.")
    parser.add_argument("--output", type=str, required=True, help="Output folder path.")
    args = parser.parse_args()

    main(args)
