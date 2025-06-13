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

def get_1d_cross_section_lenght(vectx,vecty,cellSizeX,cellSizeY):
    # General equation of the line ax+by+c=0
    a = vecty
    b = -vectx
    c1 = 0
    if vectx/abs(vectx) + vecty/abs(vecty) == 0:
        c2 = get_C_for_line_forcing_point(a,b, cellSizeX/2, cellSizeY/2) # upper-right corner C
        c3 = get_C_for_line_forcing_point(a,b, -cellSizeX/2, -cellSizeY/2) # bottom-left corner C
    else:
        c2 = get_C_for_line_forcing_point(a,b, -cellSizeX/2, cellSizeY/2) # upper-left corner C
        c3 = get_C_for_line_forcing_point(a,b, cellSizeX/2, -cellSizeY/2) # bottom-right corner C
    
    dist = distance_between_parallel_lines(a,b,c2,c3)
    return dist
