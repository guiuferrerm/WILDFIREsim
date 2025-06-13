import math
from utils import math_and_geometry_tools

def lon_deg_to_meters(lat, lon_deg, earth_radius):
    # return units depends only on earth radius units.
    # input angles in deg
    return math.cos(math_and_geometry_tools.deg_to_rad(lat)) * earth_radius * math_and_geometry_tools.deg_to_rad(lon_deg)

def lat_deg_to_meters(lat_deg, earth_radius):
    # return units depends only on earth radius units.
    # input angles in deg
    return math_and_geometry_tools.deg_to_rad(lat_deg) * earth_radius