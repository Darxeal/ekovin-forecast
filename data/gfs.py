import pathlib
import typing
import pandas as pd
import xarray as xr
import os
from datetime import datetime, timedelta
import requests
import argparse


grib_access_options = [{'level': 'surface', 'step': 'instant'}, {'level': 'meanSea', 'step': 'instant'}, 
                       {'level': 'hybrid', 'step': 'instant'}, {'level': 'atmosphere', 'step': 'instant'},
                       {'level': 'atmosphere', 'step': 'avg'}, {'level': 'surface', 'step': 'accum'}, 
                       {'level': 'surface', 'step': 'avg'}, {'level': 'planetaryBoundaryLayer', 'step': 'instant'}, 
                       {'level': 'heightAboveGround', 'step': 'max'}, {'level': 'heightAboveGround', 'step': 'min'}, 
                       {'level': 'heightAboveSea', 'step': 'instant'}, {'level': 'atmosphereSingleLayer', 'step': 'instant'}, 
                       {'level': 'atmosphereSingleLayer', 'step': 'avg'}, {'level': 'lowCloudLayer', 'step': 'instant'}, 
                       {'level': 'lowCloudLayer', 'step': 'avg'}, {'level': 'middleCloudLayer', 'step': 'instant'}, 
                       {'level': 'middleCloudLayer', 'step': 'avg'}, {'level': 'highCloudLayer', 'step': 'instant'}, 
                       {'level': 'highCloudLayer', 'step': 'avg'}, {'level': 'cloudCeiling', 'step': 'instant'}, 
                       {'level': 'convectiveCloudBottom', 'step': 'instant'}, {'level': 'lowCloudBottom', 'step': 'avg'}, 
                       {'level': 'middleCloudBottom', 'step': 'avg'}, {'level': 'highCloudBottom', 'step': 'avg'}, 
                       {'level': 'convectiveCloudTop', 'step': 'instant'}, {'level': 'lowCloudTop', 'step': 'avg'}, 
                       {'level': 'middleCloudTop', 'step': 'avg'}, {'level': 'highCloudTop', 'step': 'avg'}, 
                       {'level': 'convectiveCloudLayer', 'step': 'instant'}, {'level': 'boundaryLayerCloudLayer', 'step': 'avg'}, 
                       {'level': 'nominalTop', 'step': 'avg'}, {'level': 'heightAboveGroundLayer', 'step': 'instant'}, 
                       {'level': 'tropopause', 'step': 'instant'}, {'level': 'maxWind', 'step': 'instant'}, 
                       {'level': 'isothermZero', 'step': 'instant'}, {'level': 'highestTroposphericFreezing', 'step': 'instant'}, 
                       {'level': 'pressureFromGroundLayer', 'step': 'instant'}, {'level': 'sigma', 'step': 'instant'}]
                       
                       
def to_dataframe(grib_path: str, lat_min: float, lat_max: float, lon_min: float, lon_max: float):
    dfs = []
    reference_length = None
    for access_option in grib_access_options:
        try:
            backend_kwards = {'filter_by_keys': {'typeOfLevel': access_option['level'], 'stepType': access_option['step']}}
            grib_ds = xr.open_dataset(grib_path, engine='cfgrib', backend_kwargs=backend_kwards, errors="ignore")
            grib_ds['longitude'] = ((grib_ds['longitude'] + 180) % 360) - 180
            grib_ds_filtered = grib_ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
            df = grib_ds_filtered.to_dataframe()
        except Exception as e:
            print(f'Access level {access_option} could not be processed: {e}') 
            continue 
        df = df.drop(['time', 'step', 'valid_time'], axis=1)
        df.columns = [f"{access_option['level']}_{access_option['step']}_{col}" for col in df.columns]

        if reference_length is None:
            reference_length = len(df)
            
        if len(df) == reference_length:
            dfs.append(df)

    df_final = pd.concat(dfs, axis=1) 
    return df_final


def download_dataframe(
    current_date: datetime,
    lead_time_hours: int,
    source: typing.Literal['rda', 'nomads'],
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    tmp_file_path: str = "tmp.grib2",
):
    current_date_str = current_date.strftime("%Y%m%d%H")

    if source == "rda":
        url = f"https://data.rda.ucar.edu/ds084.1/{current_date_str[:4]}/{current_date_str[:8]}/" \
        f"gfs.0p25.{current_date_str}.f{lead_time_hours:03}.grib2"
        params = {}
    elif source == "nomads":
        url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        params = {
            "file": f"gfs.t{current_date:%H}z.pgrb2.0p25.f{lead_time_hours:03}",
            "all_lev": "on",
            "all_var": "on",
            "subregion": "",
            "leftlon": lon_min,
            "rightlon": lon_max,
            "toplat": lat_max,
            "bottomlat": lat_min,
            "dir": f"/gfs.{current_date:%Y%m%d}/{current_date:%H}/atmos",
        }
    else:
        raise Exception(f"Unsupported source: {source}")
    
    response = requests.get(url, params)

    if response.status_code == 200:
        with open(tmp_file_path, "wb") as f:
            f.write(response.content)
        
        df = to_dataframe(tmp_file_path, lat_min, lat_max, lon_min, lon_max)
        df['datetime'] = current_date

        # remove temporary files
        folder = os.path.dirname(os.path.abspath(tmp_file_path))
        for f in os.listdir(folder):
            if 'idx' in f and os.path.basename(tmp_file_path) in f:
                os.remove(os.path.join(folder, f))
        os.remove(tmp_file_path)

        return df            

    else:
        raise Exception(f"Failed to download {url} {params}, status code {response.status_code}")



def main(args):
    lon_min, lon_max = args.longitude_min_max
    lat_min, lat_max = args.latitude_min_max
    start_date = datetime.strptime(args.start_date, "%Y%m%d")
    end_date = datetime.strptime(args.end_date, "%Y%m%d")
    cols_to_include = None
    output_csv_path = os.path.join(
        args.output_folder_path, 
        f"{args.start_date}_{args.end_date}_{args.lead_time}_{args.frequency}.csv"
    )

    current_date = start_date
    while current_date <= end_date:
        try: 
            df = download_dataframe(current_date, args.lead_time, args.source, lat_min, lat_max, lon_min, lon_max, tmp_file_path=f"tmp{args.lead_time}.grib2")

            if not cols_to_include:
                cols_to_include = list(df.columns)
            
            for col in cols_to_include:
                if col not in df.columns:
                    df[col] = pd.NA
            df = df[cols_to_include] 

            # Append to CSV; check if file exists to decide on writing headers
            file_exists = os.path.isfile(output_csv_path)
            df.to_csv(output_csv_path, mode='a', header=not file_exists)

            print(f"Downloaded {current_date}")

        except Exception as ex:
            print(f"Failed to download {current_date}: {ex}")

        current_date += timedelta(hours=args.frequency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GFS Download Script")

    parser.add_argument("--start_date", type=str, required=True,
                        help="Start date in YYYYMMDD format")
    parser.add_argument("--end_date", type=str, required=True,
                        help="End date in YYYYMMDD format")
    parser.add_argument("--lead_time", type=int, required=True,
                        help="Forecast lead time in hours")
    parser.add_argument("--frequency", type=int, required=True,
                        help="Forecast frequency in hours")
    parser.add_argument("--latitude_min_max", nargs="+", type=int, default=[55, 45],
                        help="The minimum and maximum latitude of the selected area.")
    parser.add_argument("--longitude_min_max", nargs="+", type=int, default=[8, 22], 
                        help="The minimum and maximum longitude of the selected area.")
    parser.add_argument("--output_folder_path", type=str, required=True,
                        help="Output csv folder path.")
    parser.add_argument("--source", choices=["rda", "nomads"], default="ucar",
                        help="Data server to download from. Use `rda` for historical data, `nomads` for realtime.")
    args = parser.parse_args()

    main(args)