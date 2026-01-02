"""
GPT2w - Pure Python Implementation
No compilation needed! Just download gpt2_1wA.grd and run.

Based on the official GPT2w model by Böhm et al. (2015)
Reference: https://link.springer.com/article/10.1007/s10291-014-0403-7
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import os


@dataclass
class GPT2wResult:
    """Results from GPT2w computation"""
    pressure_hPa: float
    temperature_C: float
    temp_lapse_rate_K_per_km: float
    mean_temp_K: float
    water_vapor_pressure_hPa: float
    ah_hydrostatic: float
    aw_wet: float
    lambda_factor: float
    geoid_undulation_m: float
    zenith_hydrostatic_delay_m: float
    zenith_wet_delay_m: float
    zenith_total_delay_m: float


class GPT2w:
    """GPT2w tropospheric model"""

    def __init__(self, grid_file: Optional[str] = None):
        """
        Initialize GPT2w model by loading grid file.

        Parameters:
        -----------
        grid_file : str, optional
            Path to gpt2_1wA.grd file. If None, looks for it in the same
            directory as this Python file.
        """
        self.grid_loaded = False

        # If no grid file specified, look in the same directory as this module
        if grid_file is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            grid_file = os.path.join(module_dir, "gpt2_1w.grd")

        self.grid_file = grid_file

        # Grid dimensions: 180 x 360 = 64800 points
        # Latitude: -90 to 90 (1 degree spacing)
        # Longitude: 0 to 360 (1 degree spacing)
        self.nlat = 180
        self.nlon = 360
        self.npoints = self.nlat * self.nlon

        # Grid data arrays - each parameter has 5 values:
        # [0]: mean, [1]: annual cos, [2]: annual sin,
        # [3]: semi-annual cos, [4]: semi-annual sin
        self.pgrid = None  # Pressure (Pa)
        self.Tgrid = None  # Temperature (K)
        self.Qgrid = None  # Specific humidity (kg/kg)
        self.dTgrid = None  # Temperature lapse rate (K/m)
        self.u = None  # Geoid undulation (m)
        self.Hs = None  # Orthometric height (m)
        self.ahgrid = None  # Hydrostatic mapping coef
        self.awgrid = None  # Wet mapping coef
        self.lagrid = None  # Water vapor decrease factor
        self.Tmgrid = None  # Mean temperature (K)

        if os.path.exists(grid_file):
            self.load_grid()

    def load_grid(self) -> bool:
        """Load GPT2w grid file"""
        try:
            print(f"Loading GPT2w grid from {self.grid_file}...")

            # Initialize arrays
            self.pgrid = np.zeros((self.npoints, 5))
            self.Tgrid = np.zeros((self.npoints, 5))
            self.Qgrid = np.zeros((self.npoints, 5))
            self.dTgrid = np.zeros((self.npoints, 5))
            self.u = np.zeros(self.npoints)
            self.Hs = np.zeros(self.npoints)
            self.ahgrid = np.zeros((self.npoints, 5))
            self.awgrid = np.zeros((self.npoints, 5))
            self.lagrid = np.zeros((self.npoints, 5))
            self.Tmgrid = np.zeros((self.npoints, 5))

            with open(self.grid_file, 'r') as f:
                # Skip header
                f.readline()

                for n in range(self.npoints):
                    line = f.readline()
                    values = list(map(float, line.split()))

                    if len(values) < 44:
                        raise ValueError(f"Invalid grid file format at line {n + 2}")

                    # Parse grid values
                    self.pgrid[n] = values[2:7]
                    self.Tgrid[n] = values[7:12]
                    self.Qgrid[n] = np.array(values[12:17]) / 1000.0
                    self.dTgrid[n] = np.array(values[17:22]) / 1000.0
                    self.u[n] = values[22]
                    self.Hs[n] = values[23]
                    self.ahgrid[n] = np.array(values[24:29]) / 1000.0
                    self.awgrid[n] = np.array(values[29:34]) / 1000.0
                    self.lagrid[n] = values[34:39]
                    self.Tmgrid[n] = values[39:44]

            self.grid_loaded = True
            print(f"Grid loaded successfully: {self.npoints} points")
            return True

        except Exception as e:
            print(f"Error loading grid: {e}")
            return False

    def compute(self, mjd: float, lat_deg: float, lon_deg: float,
                height_m: float, time_variation: bool = True) -> GPT2wResult:
        """
        Compute GPT2w tropospheric parameters.

        Parameters:
        -----------
        mjd : float
            Modified Julian Date
        lat_deg : float
            Latitude in degrees [-90, 90]
        lon_deg : float
            Longitude in degrees [-180, 180] or [0, 360]
        height_m : float
            Ellipsoidal height in meters
        time_variation : bool
            True = with annual/semiannual terms, False = static

        Returns:
        --------
        GPT2wResult object with all parameters
        """
        if not self.grid_loaded:
            raise RuntimeError("Grid not loaded. Call load_grid() first.")

        # Convert to radians
        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)

        # Ensure longitude is positive [0, 360]
        if lon_deg < 0:
            plon = lon_deg + 360.0
        else:
            plon = lon_deg

        # Transform to polar distance in degrees
        ppod = 90.0 - lat_deg

        # Find grid indices
        ipod = int(np.floor(ppod + 1.0))
        ilon = int(np.floor(plon + 1.0))

        # Normalized differences
        diffpod = ppod - (ipod - 0.5)
        difflon = plon - (ilon - 0.5)

        # Handle edge cases
        if ipod == 181:
            ipod = 180
        if ilon == 361:
            ilon = 1
        if ilon == 0:
            ilon = 360

        # Calculate time factors
        dmjd1 = mjd - 51544.5  # Days since J2000

        if time_variation:
            cosfy = np.cos(dmjd1 / 365.25 * 2.0 * np.pi)
            coshy = np.cos(dmjd1 / 365.25 * 4.0 * np.pi)
            sinfy = np.sin(dmjd1 / 365.25 * 2.0 * np.pi)
            sinhy = np.sin(dmjd1 / 365.25 * 4.0 * np.pi)
        else:
            cosfy = coshy = sinfy = sinhy = 0.0

        # Determine if bilinear interpolation is needed
        use_bilinear = (ppod > 0.5) and (ppod < 179.5)

        if not use_bilinear:
            # Nearest neighbor
            ix = (ipod - 1) * 360 + ilon - 1
            result = self._compute_point(ix, height_m, lat,
                                         cosfy, sinfy, coshy, sinhy)
        else:
            # Bilinear interpolation
            ipod1 = ipod + int(np.sign(diffpod))
            ilon1 = ilon + int(np.sign(difflon))

            if ilon1 == 361:
                ilon1 = 1
            if ilon1 == 0:
                ilon1 = 360

            # Four corner indices
            indx = [
                (ipod - 1) * 360 + ilon - 1,
                (ipod1 - 1) * 360 + ilon - 1,
                (ipod - 1) * 360 + ilon1 - 1,
                (ipod1 - 1) * 360 + ilon1 - 1
            ]

            # Compute values at four corners
            results = []
            for idx in indx:
                results.append(self._compute_point(idx, height_m, lat,
                                                   cosfy, sinfy, coshy, sinhy))

            # Bilinear interpolation weights
            dnpod1 = abs(diffpod)
            dnpod2 = 1.0 - dnpod1
            dnlon1 = abs(difflon)
            dnlon2 = 1.0 - dnlon1

            # Interpolate
            result = {}
            for key in results[0].keys():
                R1 = dnpod2 * results[0][key] + dnpod1 * results[1][key]
                R2 = dnpod2 * results[2][key] + dnpod1 * results[3][key]
                result[key] = dnlon2 * R1 + dnlon1 * R2

        # Calculate zenith delays
        zhd = self._saasthyd(result['p'], lat, height_m)
        zwd = self._asknewet(result['e'], result['Tm'], result['la'])

        return GPT2wResult(
            pressure_hPa=result['p'],
            temperature_C=result['T'],
            temp_lapse_rate_K_per_km=result['dT'],
            mean_temp_K=result['Tm'],
            water_vapor_pressure_hPa=result['e'],
            ah_hydrostatic=result['ah'],
            aw_wet=result['aw'],
            lambda_factor=result['la'],
            geoid_undulation_m=result['undu'],
            zenith_hydrostatic_delay_m=zhd,
            zenith_wet_delay_m=zwd,
            zenith_total_delay_m=zhd + zwd
        )

    def _compute_point(self, ix: int, height_m: float, lat: float,
                       cosfy: float, sinfy: float, coshy: float, sinhy: float) -> dict:
        """Compute values at a single grid point"""

        # Constants
        gm = 9.80665  # Mean gravity (m/s^2)
        dMtr = 28.965e-3  # Molar mass of dry air (kg/mol)
        Rg = 8.3143  # Universal gas constant (J/K/mol)

        # Geoid undulation
        undu = self.u[ix]

        # Orthometric height
        hgt = height_m - undu

        # Interpolate parameters with time variation
        T0 = (self.Tgrid[ix, 0] +
              self.Tgrid[ix, 1] * cosfy + self.Tgrid[ix, 2] * sinfy +
              self.Tgrid[ix, 3] * coshy + self.Tgrid[ix, 4] * sinhy)

        p0 = (self.pgrid[ix, 0] +
              self.pgrid[ix, 1] * cosfy + self.pgrid[ix, 2] * sinfy +
              self.pgrid[ix, 3] * coshy + self.pgrid[ix, 4] * sinhy)

        Q = (self.Qgrid[ix, 0] +
             self.Qgrid[ix, 1] * cosfy + self.Qgrid[ix, 2] * sinfy +
             self.Qgrid[ix, 3] * coshy + self.Qgrid[ix, 4] * sinhy)

        dT = (self.dTgrid[ix, 0] +
              self.dTgrid[ix, 1] * cosfy + self.dTgrid[ix, 2] * sinfy +
              self.dTgrid[ix, 3] * coshy + self.dTgrid[ix, 4] * sinhy)

        ah = (self.ahgrid[ix, 0] +
              self.ahgrid[ix, 1] * cosfy + self.ahgrid[ix, 2] * sinfy +
              self.ahgrid[ix, 3] * coshy + self.ahgrid[ix, 4] * sinhy)

        aw = (self.awgrid[ix, 0] +
              self.awgrid[ix, 1] * cosfy + self.awgrid[ix, 2] * sinfy +
              self.awgrid[ix, 3] * coshy + self.awgrid[ix, 4] * sinhy)

        la = (self.lagrid[ix, 0] +
              self.lagrid[ix, 1] * cosfy + self.lagrid[ix, 2] * sinfy +
              self.lagrid[ix, 3] * coshy + self.lagrid[ix, 4] * sinhy)

        Tm = (self.Tmgrid[ix, 0] +
              self.Tmgrid[ix, 1] * cosfy + self.Tmgrid[ix, 2] * sinfy +
              self.Tmgrid[ix, 3] * coshy + self.Tmgrid[ix, 4] * sinhy)

        # Height correction
        redh = hgt - self.Hs[ix]

        # Temperature at station height
        T = T0 + dT * redh - 273.15  # Convert to Celsius

        # Temperature lapse rate in K/km
        dT_km = dT * 1000.0

        # Virtual temperature
        Tv = T0 * (1.0 + 0.6077 * Q)
        c = gm * dMtr / (Rg * Tv)

        # Pressure at station height
        p = (p0 * np.exp(-c * redh)) / 100.0  # Convert to hPa

        # Water vapor pressure
        e0 = Q * p0 / (0.622 + 0.378 * Q) / 100.0
        e = e0 * ((100.0 * p / p0) ** (la + 1.0))

        return {
            'p': p,
            'T': T,
            'dT': dT_km,
            'Tm': Tm,
            'e': e,
            'ah': ah,
            'aw': aw,
            'la': la,
            'undu': undu
        }

    @staticmethod
    def _saasthyd(p: float, lat: float, height_m: float) -> float:
        """Calculate zenith hydrostatic delay (Saastamoinen)"""
        f = 1.0 - 0.00266 * np.cos(2.0 * lat) - 0.00000028 * height_m
        zhd = 0.0022768 * p / f
        return zhd

    @staticmethod
    def _asknewet(e: float, Tm: float, lambda_factor: float) -> float:
        """Calculate zenith wet delay (Askne & Nordius)"""
        k1 = 77.604
        k2 = 64.79
        k2p = k2 - k1 * 18.0152 / 28.9644
        k3 = 377600.0
        gm = 9.80665
        dMtr = 28.965e-3
        R = 8.3143
        Rd = R / dMtr

        zwd = 1.0e-6 * (k2p + k3 / Tm) * Rd / (lambda_factor + 1.0) / gm * e
        return zwd

    def compute_slant_delay(self, mjd: float, lat_deg: float, lon_deg: float,
                            height_m: float, elevation_deg: float,
                            time_variation: bool = True) -> dict:
        """
        Compute slant delays for given elevation angle.

        Parameters:
        -----------
        mjd : float
            Modified Julian Date
        lat_deg : float
            Latitude in degrees
        lon_deg : float
            Longitude in degrees
        height_m : float
            Ellipsoidal height in meters
        elevation_deg : float
            Elevation angle in degrees
        time_variation : bool
            Use time variation (default: True)

        Returns:
        --------
        Dictionary with slant delays and mapping functions
        """
        # Get zenith parameters
        result = self.compute(mjd, lat_deg, lon_deg, height_m, time_variation)

        # Convert to radians
        lat = np.deg2rad(lat_deg)
        zd = np.deg2rad(90.0 - elevation_deg)  # Zenith distance

        # Calculate mapping functions
        mfh, mfw = self._vmf1_ht(result.ah_hydrostatic, result.aw_wet,
                                 mjd, lat, height_m, zd)

        return {
            'zenith_hydrostatic_delay_m': result.zenith_hydrostatic_delay_m,
            'zenith_wet_delay_m': result.zenith_wet_delay_m,
            'mapping_function_hydro': mfh,
            'mapping_function_wet': mfw,
            'slant_hydrostatic_delay_m': result.zenith_hydrostatic_delay_m * mfh,
            'slant_wet_delay_m': result.zenith_wet_delay_m * mfw,
            'slant_total_delay_m': (result.zenith_hydrostatic_delay_m * mfh +
                                    result.zenith_wet_delay_m * mfw),
            'elevation_deg': elevation_deg
        }

    @staticmethod
    def _vmf1_ht(ah: float, aw: float, dmjd: float, dlat: float,
                 ht: float, zd: float) -> Tuple[float, float]:
        """Vienna Mapping Function 1 with height correction"""
        pi = np.pi

        # Day of year (reference: Jan 28)
        doy = dmjd - 44239.0 + 1.0 - 28.0

        # Hydrostatic
        bh = 0.0029
        c0h = 0.062

        if dlat < 0:  # Southern hemisphere
            phh = pi
            c11h = 0.007
            c10h = 0.002
        else:  # Northern hemisphere
            phh = 0.0
            c11h = 0.005
            c10h = 0.001

        ch = c0h + ((np.cos(doy / 365.25 * 2.0 * pi + phh) + 1.0) *
                    c11h / 2.0 + c10h) * (1.0 - np.cos(dlat))

        sine = np.sin(pi / 2.0 - zd)
        beta = bh / (sine + ch)
        gamma = ah / (sine + beta)
        topcon = 1.0 + ah / (1.0 + bh / (1.0 + ch))
        vmf1h = topcon / (sine + gamma)

        # Height correction
        a_ht = 2.53e-5
        b_ht = 5.49e-3
        c_ht = 1.14e-3
        hs_km = ht / 1000.0

        beta = b_ht / (sine + c_ht)
        gamma = a_ht / (sine + beta)
        topcon = 1.0 + a_ht / (1.0 + b_ht / (1.0 + c_ht))
        ht_corr_coef = 1.0 / sine - topcon / (sine + gamma)
        ht_corr = ht_corr_coef * hs_km
        vmf1h += ht_corr

        # Wet
        bw = 0.00146
        cw = 0.04391
        beta = bw / (sine + cw)
        gamma = aw / (sine + beta)
        topcon = 1.0 + aw / (1.0 + bw / (1.0 + cw))
        vmf1w = topcon / (sine + gamma)

        return vmf1h, vmf1w


# Example usage
if __name__ == "__main__":
    from datetime import datetime

    # Initialize model
    gpt2w = GPT2w("gpt2_1w.grd")


    # Convert datetime to MJD
    def datetime_to_mjd(dt):
        jd = dt.toordinal() + 1721424.5
        mjd = jd - 2400000.5
        return mjd


    # Example: Vienna, August 2, 2012
    dt = datetime(2012, 8, 2, 12, 0, 0)
    mjd = datetime_to_mjd(dt)

    lat = 48.20
    lon = 16.37
    height = 156.0

    print("=" * 60)
    print("GPT2w Tropospheric Parameters")
    print("=" * 60)
    print(f"Location: {lat}°N, {lon}°E, {height}m")
    print(f"Date: {dt}")
    print(f"MJD: {mjd:.1f}")
    print("=" * 60)

    result = gpt2w.compute(mjd, lat, lon, height)

    print(f"Pressure: {result.pressure_hPa:.2f} hPa")
    print(f"Temperature: {result.temperature_C:.2f} °C")
    print(f"Temperature lapse rate: {result.temp_lapse_rate_K_per_km:.2f} K/km")
    print(f"Mean temperature: {result.mean_temp_K:.2f} K")
    print(f"Water vapor pressure: {result.water_vapor_pressure_hPa:.2f} hPa")
    print(f"Zenith hydrostatic delay: {result.zenith_hydrostatic_delay_m:.4f} m")
    print(f"Zenith wet delay: {result.zenith_wet_delay_m:.4f} m")
    print(f"Zenith total delay: {result.zenith_total_delay_m:.4f} m")

    print("\n" + "=" * 60)
    print("Slant Delay at 10° Elevation")
    print("=" * 60)

    slant = gpt2w.compute_slant_delay(mjd, lat, lon, height, 10.0)
    print(f"Slant total delay: {slant['slant_total_delay_m']:.4f} m")
    print(f"Mapping function (hydro): {slant['mapping_function_hydro']:.4f}")
    print(f"Mapping function (wet): {slant['mapping_function_wet']:.4f}")