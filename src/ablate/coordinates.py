"""
# Coordinate functions
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


# Constants defined by the World Geodetic System 1984 (WGS84)
WGS84_a = 6378.137 * 1e3
WGS84_b = 6356.7523142 * 1e3
WGS84_esq = 6.69437999014 * 0.001
WGS84_e1sq = 6.73949674228 * 0.001


def geodetic2ecef(lat, lon, alt):
    """Convert WGS84 geodetic coordinates to ECEF coordinates.

    [^1] J. Zhu, "Conversion of Earth-centered Earth-fixed coordinates to geodetic coordinates,"
        IEEE Transactions on Aerospace and Electronic Systems, vol. 30, pp. 957-961, 1994.

    lat : float
        Geographic latitude [deg]
    lon : float
        Geographic longitude [deg]
    alt : float
        Geographic altitude [deg]

    :rtype: np.ndarray
    :return: [x, y, z] in ECEF coordinates

    """
    lat, lon = np.radians(lat), np.radians(lon)
    xi = np.sqrt(1 - WGS84_esq * np.sin(lat) ** 2)
    x = (WGS84_a / xi + alt) * np.cos(lat) * np.cos(lon)
    y = (WGS84_a / xi + alt) * np.cos(lat) * np.sin(lon)
    z = (WGS84_a / xi * (1 - WGS84_esq) + alt) * np.sin(lat)
    return np.array([x, y, z])


def ecef2geodetic(x, y, z):
    """Convert ECEF coordinates to WGS84 geodetic coordinates.

    :param float x: Position along prime meridian [m]
    :param float y: Position along prime meridian + 90 degrees [m]
    :param float z: Position along earth rotation axis [m]

    :rtype: np.ndarray
    :return: [lat [deg], lon [deg], alt [m]] in WGS84 geodetic coordinates

    **References:**

        * J. Zhu, "Conversion of Earth-centered Earth-fixed coordinates to geodetic coordinates," IEEE Transactions on Aerospace and Electronic Systems, vol. 30, pp. 957-961, 1994.

    """
    r = np.sqrt(x * x + y * y)
    if r < 1e-9:
        h = np.abs(z) - WGS84_b
        lat = np.sign(z) * np.pi / 2
        lon = 0.0
    else:
        Esq = WGS84_a * WGS84_a - WGS84_b * WGS84_b
        F = 54 * WGS84_b * WGS84_b * z * z
        G = r * r + (1 - WGS84_esq) * z * z - WGS84_esq * Esq
        C = (WGS84_esq * WGS84_esq * F * r * r) / (np.power(G, 3))
        S = np.cbrt(1 + C + np.sqrt(C * C + 2 * C))
        P = F / (3 * np.power((S + 1 / S + 1), 2) * G * G)
        Q = np.sqrt(1 + 2 * WGS84_esq * WGS84_esq * P)
        r_0 = -(P * WGS84_esq * r) / (1 + Q) + np.sqrt(
            0.5 * WGS84_a * WGS84_a * (1 + 1.0 / Q)
            - P * (1 - WGS84_esq) * z * z / (Q * (1 + Q))
            - 0.5 * P * r * r
        )
        U = np.sqrt(np.power((r - WGS84_esq * r_0), 2) + z * z)
        V = np.sqrt(np.power((r - WGS84_esq * r_0), 2) + (1 - WGS84_esq) * z * z)
        Z_0 = WGS84_b * WGS84_b * z / (WGS84_a * V)
        h = U * (1 - WGS84_b * WGS84_b / (WGS84_a * V))
        lat = np.arctan((z + WGS84_e1sq * Z_0) / r)
        lon = np.arctan2(y, x)

    return np.array([np.degrees(lat), np.degrees(lon), h])


def enu2ecef(lat, lon, alt, e, n, u):
    """NEU (north/east/up) to ECEF coordinate system conversion. Degrees are used."""
    return ned2ecef(lat, lon, alt, n, e, -u)


def ned2ecef(lat, lon, alt, n, e, d):
    """NED (north/east/down) to ECEF coordinate system conversion. Degrees are used."""
    lat, lon = np.radians(lat), np.radians(lon)
    mx = np.array(
        [
            [-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
            [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
            [0, np.cos(lat), np.sin(lat)],
        ]
    )
    enu = np.array([e, n, -1.0 * d])
    res = np.dot(mx, enu)
    return res


def ecef2ned(lat, lon, alt, x, y, z):
    """NED (east,north,up) from ECEF coordinate system conversion."""
    lat, lon = np.radians(lat), np.radians(lon)
    mx = np.array(
        [
            [-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
            [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
            [0, np.cos(lat), np.sin(lat)],
        ]
    )
    enu = np.array([x, y, z])
    res = np.dot(np.linalg.inv(mx), enu)
    return res


def cart_to_azel(vec):
    """Convert from Cartesian coordinates to spherical in a degrees east of north and elevation fashion"""
    x = vec[0]
    y = vec[1]
    z = vec[2]
    r_ = np.sqrt(x**2 + y**2)
    if r_ < 1e-9:
        el = np.sign(z) * np.pi * 0.5
        az = 0.0
    else:
        el = np.arctan(z / r_)
        az = np.pi / 2 - np.arctan2(y, x)
    return np.degrees(az), np.degrees(el), np.sqrt(x**2 + y**2 + z**2)


def azel_to_cart(az, el, r):
    """Convert from spherical coordinates to Cartesian in a degrees east of north and elevation fashion"""
    _az = np.radians(az)
    _el = np.radians(el)
    return r * np.array([np.sin(_az) * np.cos(_el), np.cos(_az) * np.cos(_el), np.sin(_el)])


def curved_earth(s, rp, h_obs, zd, h_start):
    """Calculates the error of the height? what



    distance :code:`s` along the trajectory at :code:`h=h_start` when :code:`s=0` at :code:`h=h_obs` given that the zentith distance is :code:`zd` at :code:`h=h_obs`.

    """

    theta = np.arctan2(s * np.sin(zd), (rp + h_obs + s * np.cos(zd)))
    h = (rp + h_obs + s * np.cos(zd)) / np.cos(theta) - rp

    err = h - h_start

    return err
