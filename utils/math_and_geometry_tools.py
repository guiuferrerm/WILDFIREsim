import math

def deg_to_rad(deg):
    return deg*(math.pi/180)

def rad_to_deg(rad):
    return rad*(180/math.pi)

def distance_between_parallel_lines(a,b,c1,c2):
    # General equation of the line ax+by+c=0
    return abs(c1-c2)/math.sqrt(a**2 + b**2)

def get_C_for_line_forcing_point(a,b,forcedPointX,forcedPointY):
    # General equation of the line ax+by+c=0
    return -(a*forcedPointX+b*forcedPointY)

def get_1d_cross_section_length(vectx, vecty, cellSizeX, cellSizeY):
    # General equation of the line: ax + by + c = 0
    a = vecty
    b = -vectx

    # Handle vertical and horizontal cases
    if vectx == 0:
        return cellSizeX
    elif vecty == 0:
        return cellSizeY

    # Determine orientation to select the two corners of the cell
    sign_x = vectx / abs(vectx)
    sign_y = vecty / abs(vecty)

    if sign_x + sign_y == 0:
        # Diagonal: upper-right and bottom-left
        c2 = get_C_for_line_forcing_point(a, b, cellSizeX / 2, cellSizeY / 2)
        c3 = get_C_for_line_forcing_point(a, b, -cellSizeX / 2, -cellSizeY / 2)
    else:
        # Diagonal: upper-left and bottom-right
        c2 = get_C_for_line_forcing_point(a, b, -cellSizeX / 2, cellSizeY / 2)
        c3 = get_C_for_line_forcing_point(a, b, cellSizeX / 2, -cellSizeY / 2)

    dist = distance_between_parallel_lines(a, b, c2, c3)
    return dist
