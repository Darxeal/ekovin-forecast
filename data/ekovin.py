import argparse
import os
import pandas as pd
import requests
import io
from datetime import datetime


def download(probe_id: int, latitude: float, longitude: float, dt_start: datetime, dt_end: datetime):
    dt_start = int(datetime.timestamp(dt_start))
    dt_end = int(datetime.timestamp(dt_end))
    url = f"http://a.la-a.la/chart/data_noel.php?probe={probe_id}&t1={dt_start}&t2={dt_end}"

    response = requests.get(url).content.decode("Windows-1250")
    response_csv = "\n".join(response.split("\n")[7:])
    df = pd.read_table(io.StringIO(response_csv), sep=",")

    df.index = pd.to_datetime(df.DATUM + " " + df.CAS, dayfirst=True) # + pd.to_timedelta(df["RELATIVNI CAS"], unit="h")
    df.index.name = "datetime"
    df = df.drop(columns=["DATUM", "CAS", "RELATIVNI CAS", "STAV"])

    # df = df.dropna(axis=1, how='all')
    # df = df.loc[:, df.nunique() > 1]
    # df = df.drop(columns=[col for col in df.columns if 'rezerva' in col])

    df["latitude"] = latitude
    df["longitude"] = longitude
    df["station_index"] = probe_id
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EKOVIN Download Script")

    parser.add_argument("--probe", type=int, required=True,
                        help="Probe ID")
    parser.add_argument("--latitude", type=float,
                        help="Latitude of the probe.")
    parser.add_argument("--longitude", type=float, 
                        help="Longitude of the probe.")
    parser.add_argument("--start_datetime", type=str, required=True,
                        help="Start datetime in any ISO format.")
    parser.add_argument("--end_datetime", type=str, required=True,
                        help="End datetime in any ISO format.")
    parser.add_argument("--output_folder_path", type=str, required=True,
                        help="Output csv folder path.")
    args = parser.parse_args()

    df = download(args.probe, args.latitude, args.longitude, datetime.fromisoformat(args.start_datetime), datetime.fromisoformat(args.end_datetime))
    df.to_csv(os.path.join(args.output_folder_path, f"{args.probe}.csv"))
