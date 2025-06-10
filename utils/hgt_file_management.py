from utils import geography_tools 

def HGT_to_np_array(filepath):
    import os
    import math
    import numpy as np
    
    # require os, math, numpy
    siz = os.path.getsize(filepath)
    dim = int(math.sqrt(siz/2))
    
    assert dim*dim*2 == siz, 'Invalid file size'
    
    data = np.fromfile(filepath, np.dtype('>i2'), dim*dim).reshape((dim, dim))
    npData = data.byteswap().view(data.dtype.newbyteorder('='))  # Ensure the array is in native endianness format (if it's not already) || Convert to native endianness
    npData = npData.astype("float64")
    npData = np.where(npData<0, 0, npData)
    
    return npData

def convert_to_slice(slice_str):
    """
    Converts a slice string (e.g., '0:542', ':650', '500:', etc.) into a slice object.
    """
    if slice_str == ':':  # Special case for full slice
        return slice(None)

    parts = slice_str.split(':')

    # Handle cases like '500:', ':650', or '0:542'
    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
    step = int(parts[2]) if len(parts) > 2 and parts[2] else None

    return slice(start, stop, step)

def prepare_HGT_as_array_data(data, originN, minN, maxN, originE, minE, maxE, arcsecInterval):
    import numpy as np

    
    '''
    data is array of height data. Arranged like:
    somethingN
    |
    |
    |
    |
    |
    (originN ----- somethingE
    originE)
    
    originN, minN, maxN, originE, minE, maxE should be in degrees (lat/lon).
    Arcsec interval: how many arcsec between array elements.
    
    '''
    
    data = np.flipud(data) # flip hgt data vertically to fit numpy format (up--> down, left--> right)
    
    oneArcsecAsDegree = 1/3600
    arcsecsInDegree = 3600
    earth_radius = 6371000

    avgLat = (minN+maxN)/2
    avgLon = (minE+maxE)/2

    meterIntervalX = geography_tools.lon_deg_to_meters(avgLat, oneArcsecAsDegree, earth_radius)*arcsecInterval
    meterIntervalY = geography_tools.lat_deg_to_meters(oneArcsecAsDegree, earth_radius)*arcsecInterval
    
    # get min and max y cells
    minimumYCell = round((minN - originN)*arcsecsInDegree/arcsecInterval)
    if minimumYCell < 0:
        minimumYCell = 0
        raise IndexError("Minimum N value out of range for file")
    
    maximumYCell = round((maxN - originN)*arcsecsInDegree/arcsecInterval)
    if maximumYCell > data.shape[0]-1:
        maximumYCell = data.shape[0]-1
        raise IndexError("Maximum N value out of range for file")
    
    # get min and max x cells
    minimumXCell = round((minE - originE)*arcsecsInDegree/arcsecInterval)
    if minimumXCell < 0:
        minimumXCell = 0
        raise IndexError("Minimum E value out of range for file")
    
    maximumXCell = round((maxE - originE)*arcsecsInDegree/arcsecInterval)
    if maximumXCell > data.shape[1]-1:
        maximumXCell = data.shape[1]-1
        raise IndexError("Maximum E value out of range for file")
    
    # Prepare the slices for data wanted
    row_slice = convert_to_slice(f"{int(minimumYCell)}:{int(maximumYCell)}")
    col_slice = convert_to_slice(f"{int(minimumXCell)}:{int(maximumXCell)}")
    selectedData = data[row_slice, col_slice] # Select data wanted (Y/X remember) by cells
    
    # Create meshgrids for data: function returns the data selected + way to interpret the data. Bc numpy y works inverse to cartesian y, the meshgrid returns where should each y value really go (the np array is built to be read in cartesian way).
    X, Y = np.meshgrid(np.linspace(0, maximumXCell*meterIntervalX-minimumXCell*meterIntervalX, maximumXCell-minimumXCell), np.linspace(0, maximumYCell*meterIntervalY-minimumYCell*meterIntervalY, maximumYCell-minimumYCell))   # Create meshgrid for X, Y coordinates
    meterMeshGrid = X, Y
    X, Y = np.meshgrid(np.linspace((minimumXCell*arcsecInterval/arcsecsInDegree)+originE, (maximumXCell*arcsecInterval/arcsecsInDegree)+originE, maximumXCell-minimumXCell), np.linspace((minimumYCell*arcsecInterval/arcsecsInDegree)+originN, (maximumYCell*arcsecInterval/arcsecsInDegree)+originN, maximumYCell-minimumYCell))
    arcsecMeshGrid = X, Y

    print(meterMeshGrid)
    print(arcsecMeshGrid)
    
    return selectedData, meterMeshGrid, arcsecMeshGrid
