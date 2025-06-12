import numpy as np
from utils.npz_file_management import build_npz_file

def create_new_wfss_file(setupName, dataArrays, heightData, arcsecMeshGrid, meterMeshGrid):
    xMeshM, yMeshM = meterMeshGrid
    Xmesh, Ymesh = arcsecMeshGrid
    
    arrays = {
        "height": np.copy(heightData),
        "x_deg_mesh": np.copy(Xmesh),
        "y_deg_mesh": np.copy(Ymesh),
        "x_meter_mesh": np.copy(xMeshM),
        "y_meter_mesh": np.copy(yMeshM),
        "temperature": np.copy(dataArrays["temperature"]["array"]),
        "fuel_moisture_content": np.copy(dataArrays["fuel_moisture_content"]["array"]),
        "fuel_mass": np.copy(dataArrays["fuel_mass"]["array"]),
        "unburnable_mass": np.copy(dataArrays["unburnable_mass"]["array"]),
        "wind_x": np.copy(dataArrays["wind_x"]["array"]),
        "wind_y": np.copy(dataArrays["wind_y"]["array"]),

        "unmod_settings": {
            "cell_size_x": round(abs(xMeshM[0][-1] - xMeshM[0][0]) / (xMeshM.shape[1] - 1), 3),
            "cell_size_y": round(abs(yMeshM[:,0][-1] - yMeshM[:,0][0]) / (yMeshM.shape[0] - 1), 3),

            "array_dim_x": int(heightData.shape[1]),
            "array_dim_y": int(heightData.shape[0]),

            "water_specific_heat": 4.186,
            "water_latent_heat": 2260.0,
            "water_boiling_temp": 373.15,
        },

        "mod_settings": {
            "fuel_igniting_temp": 573.15,
            "fuel_specific_heat": 1.76,
            "fuel_calorific_value": 20900.0,

            "unburnable_specific_heat": 1.17,

            "ambient_temp": 298.15,
            "boundary_avg_mass": 30.0,
            "boundary_avg_specific_heat": 2.0,
            "boundary_avg_wind_vector_x": 0,
            "boundary_avg_wind_vector_y": 0,

            "heat_transfer_factor": 0.003,
            "slope_effect_factor": 0.2,
            "wind_effect_factor": 1,
            "fuel_burn_rate": 0.001,
            "heat_loss_factor": 0.999,
            "transfer_heat_loss_factor": 0.7,
            "burn_heat_loss_factor": 0.7,
            
        },

        "metadata": {
            "version": 1.12,
            "setup_title": setupName,
        },
    }

    file = build_npz_file(arrays)

    return file