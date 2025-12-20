from datetime import datetime, timezone
from gnss_lib_py.utils.ephemeris_downloader import _download_ephemeris
from gnss_lib_py.utils.time_conversions import datetime_to_tow,gps_millis_to_tow

import numpy as np
import gnss_lib_py as glp

xy_time=1273529463442

p,q = gps_millis_to_tow(xy_time)

print(p)
print(q)

lat, lon, alt = 37.42984154652992, -122.16946303566934, 0.
timestamp_start = datetime(year=2020, month=5, day=13, hour=12, tzinfo=timezone.utc)
timestamp_end = datetime(year=2020, month=5, day=14, hour=13, tzinfo=timezone.utc)

gps_millis = glp.datetime_to_gps_millis(np.array([timestamp_start,timestamp_end]))
sp3_path = glp.load_ephemeris(file_type="sp3",
                              gps_millis=gps_millis,
                              verbose=True)
