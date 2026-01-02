""" Script to process Tropo data and compare with GPT2w model

"""

import os
from datetime import datetime, timezone, timedelta

import numpy as np
from pathlib import Path

from GPT2w.GPT2w import GPT2w

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.utils.time_conversions import gps_datetime_to_gps_millis, datetime_to_mjd
from gnss_lib_py.utils import time_conversions

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

class Tropo(NavData):
    """
    Tropospheric delay loading
    """

    def __init__(self, input_paths):
        super().__init__()

        if isinstance(input_paths, (str, os.PathLike)):
            input_paths = [input_paths]

        self.station_id = str()
        self.station_lla = np.zeros(3, dtype=np.float64)
        gps_millis = []
        gps_weeks = []
        gps_tows = []
        ztd_m = []          # Zenith Troposheric Delay [m]

        for input_path in input_paths:
            # Initial checks for loading file
            if not isinstance(input_path, (str, os.PathLike)):
                raise TypeError("input_path must be string or path-like")
            if not os.path.exists(input_path):
                raise FileNotFoundError(input_path, "file not found")

            # Load in the file
            with open(input_path, 'r', encoding="utf-8") as infile:
                # Loop through each line
                for line in infile:
                    if "+SITE/ID" in line:
                        line = next(infile)
                        line = next(infile)
                        data = line.strip().split()

                        if not self.station_id :
                            self.station_id = data[0]
                        else:
                            # ensure station id is matching with currently assigned value
                            if data[0] != self.station_id:
                                print(f" Mis-match is detected in station id ({self.station_id}) and data ({data[0]})")

                        self.station_lla[0] = np.float64(data[5])
                        self.station_lla[1] = np.float64(data[6])
                        self.station_lla[2] = np.float64(data[7])

                    if "+TROP/SOLUTION" in line:
                        line = next(infile)  # header of data
                        line = next(infile)  # First data; Assumption is that single data line always available

                        while True:
                            data = line.strip().split()
                            curr_time = self._parse_yds_time(data[1])
                            gps_millis_timestep = time_conversions.gps_datetime_to_gps_millis(curr_time)
                            week, tow = time_conversions.datetime_to_tow(curr_time)

                            gps_millis.append(gps_millis_timestep)
                            gps_weeks.append(week)
                            gps_tows.append(tow)
                            ztd_m.append(float(data[2])/1000)           # mm to meter
                            line = next(infile)         # read next line
                            if "-TROP/SOLUTION" in line:
                                break

        self["gps_millis"] = gps_millis
        self["ztd_m"] = ztd_m
        self["gps_tow"] = gps_tows
        self["gps_week"] = gps_weeks

    @staticmethod
    def _parse_yds_time(time_str):
        """Parse Year:DayOfYear:SecondsOfDay format"""
        parts = time_str.split(':')

        year = 2000 + int(parts[0])
        day_of_year = int(parts[1])
        seconds_of_day = int(parts[2])

        # Convert seconds to hours, minutes, seconds
        hours = seconds_of_day // 3600
        minutes = (seconds_of_day % 3600) // 60
        secs = seconds_of_day % 60

        # Build datetime
        dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
            days=day_of_year - 1,
            hours=hours,
            minutes=minutes,
            seconds=secs
        )

        return dt

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = Path(__file__).resolve().parent

    input_trop_file = script_dir / "data" / "JPS1_SES_FIN_20160030000_01D_00U_KOKV_TRO"

    tropo_data = Tropo(input_trop_file)

    # Initialize GPT2w Tropo model
    gpt2w = GPT2w()


    # Example: Vienna, August 2, 2012
    dt = time_conversions.gps_millis_to_datetime(tropo_data["gps_millis"])
    mjd = datetime_to_mjd(dt)

    lat = tropo_data.station_lla[0]
    lon = tropo_data.station_lla[1]
    height = tropo_data.station_lla[2]

    print("=" * 60)
    print("GPT2w Tropospheric Parameters")
    print("=" * 60)
    print(f"Location: {lat}°N, {lon}°E, {height}m")
    # print(f"Date: {dt}")
    # print(f"MJD: {mjd:.1f}")
    print("=" * 60)

    result = gpt2w.compute(mjd, lat, lon, height)
    modeled_T = result.zenith_total_delay_m

    residual_GPT2w = tropo_data["ztd_m"] - modeled_T

    # Create figure
    p = figure(
        width=1000,
        height=700,
        title="Residual of GPT2w modeled zenith Tropospheric delay",
        x_axis_label="GPS ToW  [s]",
        y_axis_label="Residuals [m]",
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )

    # Create data source for GNSS
    gnss_source = ColumnDataSource(data=dict(
        x=tropo_data["gps_tow"],
        y=residual_GPT2w,
    ))
    # Plot residuals
    gnss_line = p.line('x', 'y', source=gnss_source,
                       line_width=2, color='blue', alpha=0.6,
                       legend_label='Residuals')
    gnss_points = p.scatter('x', 'y', source=gnss_source,
                           size=6, marker="circle", color='blue', alpha=0.8)

    # Configure legend
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    show(p)

    print("="*60)
    print(f"{' '*18} End of Tropo Analysis {' '*18}")
    print("="*60)