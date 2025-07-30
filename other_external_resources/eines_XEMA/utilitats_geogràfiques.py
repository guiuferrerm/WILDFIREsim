import math
import numpy as np

R_EARTH_KM = 6371.0088

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Distància gran-cercle entre dos punts (en km).
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R_EARTH_KM * c

def cercle_geografic(lat_centre, lon_centre, radi_km, n_punts=100):
    """
    Retorna latituds i longituds que formen un cercle geogràfic de radi_km
    al voltant del punt (lat_centre, lon_centre).
    """
    angles = np.linspace(0, 2*np.pi, n_punts)
    lats, lons = [], []
    d = radi_km / R_EARTH_KM
    lat_rad = math.radians(lat_centre)
    lon_rad = math.radians(lon_centre)

    for angle in angles:
        lat_p = math.asin(
            math.sin(lat_rad)*math.cos(d) + math.cos(lat_rad)*math.sin(d)*math.cos(angle)
        )
        lon_p = lon_rad + math.atan2(
            math.sin(angle)*math.sin(d)*math.cos(lat_rad),
            math.cos(d) - math.sin(lat_rad)*math.sin(lat_p)
        )
        lats.append(math.degrees(lat_p))
        lons.append(math.degrees(lon_p))
    return lats, lons
