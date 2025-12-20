import numpy as np
import requests
from pathlib import Path

from gnss_lib_py.parsers.sp3 import Sp3
import gnss_lib_py as glp

glp.make_dir("../data_xyz")

url = "https://raw.githubusercontent.com/Stanford-NavLab/gnss_lib_py/main/data/unit_test/sp3/COD0MGXFIN_20211180000_01D_05M_ORB.SP3"
out = Path("../data/COD0MGXFIN_20211180000_01D_05M_ORB.SP3")

out.parent.mkdir(parents=True, exist_ok=True)

r = requests.get(url)
r.raise_for_status()
out.write_bytes(r.content)



# Specify .sp3 file path to extract precise ephemerides
sp3_path = "../data/COD0MGXFIN_20211180000_01D_05M_ORB.SP3"



