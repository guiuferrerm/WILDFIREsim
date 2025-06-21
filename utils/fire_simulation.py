import numpy as np
import math
from scipy.signal import convolve2d
from utils.cache_config import cache
from utils import math_and_geometry_tools
from utils import grid_and_simulation_tools

class SimGrid:
    # GRID SETUP, CREATION AND RESET
    def __init__(self):
        # Arrays
        self.cellSizeX = 0 # in meters
        self.cellSizeY = 0 # in meters
        self.cellArea = 0
        self.gridSizeX = 0
        self.gridSizeY = 0
        self.cellEffectRadius = 0 # in meters

        # Geography
        self.cellHeight = None

        # initial conditions
        self.initialConditions = {
            'cellTemperature': None,
        }

        # Fuel
        self.fuelIgnitingTemp = 0
        self.fuelSpecificHeat = 0
        self.fuelCalorificValue = 0
        self.fuelMass = None
        self.fuelTemperature = None
        self.fuelThermalEnergy = None
        self.fuelBurnRate = 0

        # Water
        self.waterSpecificHeat = 0
        self.waterLatentHeat = 0
        self.waterBoilingTemp = 0
        self.waterMass = None
        self.waterTemperature = None
        self.waterThermalEnergy = None

        # Unburnable Material
        self.unburnableSpecificHeat = 0
        self.unburnableMass = None
        self.unburnableTemperature = None
        self.unburnableThermalEnergy = None

        # Cell-wise Thermodynamics
        self.cellTotalMass = None
        self.cellSpecificHeat = None
        self.cellTemperature = None
        self.cellThermalE = None
        self.fireIntensity = None

        # Boundary
        self.ambientTemp = 0
        self.boundaryMass = 0
        self.boundaryCe = 0
        self.boundaryWindVector = None  # Will be np.array([y, x])

        # Wind
        self.windField = None  # 3D array (Y, X, 2)

        # Fire Spread Constants
        self.heatTransferRate = 0
        self.slopeEffectFactor = 0
        self.windEffectFactor = 0
        self.heatLossFactor = 0
        self.transferHeatLossFactor = 0
        self.burnHeatLossFactor = 0

        self.neighbors_kernel = None
        self.neighbors_vector_list_array = None
        self.heightGradients = None

    def reset(self):
        self.__init__()

    def setup(self,
            heightArray, temperatureArray, fuelMoistureArray, fuelMassArray, unburnableMassArray, windXArray, windYArray,
            unmod_settings, mod_settings, metadata,
            initial_igniting_cells):

        # Grid structure
        self.cellSizeX = float(unmod_settings["cell_size_x"])
        self.cellSizeY = float(unmod_settings["cell_size_y"])
        self.cellArea = self.cellSizeX * self.cellSizeY
        self.gridSizeX = int(unmod_settings["array_dim_x"])
        self.gridSizeY = int(unmod_settings["array_dim_y"])
        self.cellEffectRadius = mod_settings["cell_effect_radius"]

        self.cellHeight = np.copy(heightArray.astype(np.int32))

        # Core arrays
        self.fuelMass = np.copy(fuelMassArray)
        self.unburnableMass = np.copy(unburnableMassArray)
        self.waterMass = np.copy(fuelMassArray * (fuelMoistureArray / 100.0))

        self.windField = np.zeros((self.gridSizeY, self.gridSizeX, 2))
        self.windField[:, :, 1] = np.copy(windXArray)
        self.windField[:, :, 0] = np.copy(windYArray)

        self.initialConditions['cellTemperature'] = np.copy(temperatureArray)
        self.cellTemperature = np.copy(temperatureArray)
        self.fuelTemperature = np.copy(temperatureArray)
        self.waterTemperature = np.copy(temperatureArray)
        self.unburnableTemperature = np.copy(temperatureArray)

        # Constants: unmodified
        self.waterSpecificHeat = unmod_settings["water_specific_heat"]
        self.waterLatentHeat = unmod_settings["water_latent_heat"]
        self.waterBoilingTemp = unmod_settings["water_boiling_temp"]

        # Constants: modified
        self.fuelIgnitingTemp = mod_settings["fuel_igniting_temp"]
        self.fuelSpecificHeat = mod_settings["fuel_specific_heat"]
        self.fuelCalorificValue = mod_settings["fuel_calorific_value"]
        self.unburnableSpecificHeat = mod_settings["unburnable_specific_heat"]
        self.ambientTemp = mod_settings["ambient_temp"]

        self.boundaryMass = mod_settings["boundary_avg_mass"]
        self.boundaryCe = mod_settings["boundary_avg_specific_heat"]
        self.boundaryWindVector = np.array([
            mod_settings["boundary_avg_wind_vector_x"],
            mod_settings["boundary_avg_wind_vector_y"]
        ])

        self.heatTransferRate = mod_settings["heat_transfer_rate"] # Kj/(s*m*K)
        self.slopeEffectFactor = mod_settings["slope_effect_factor"]
        self.windEffectFactor = mod_settings["wind_effect_factor"]
        self.fuelBurnRate = mod_settings["fuel_burn_rate"]
        self.heatLossFactor = mod_settings["heat_loss_factor"]
        self.transferHeatLossFactor = mod_settings["transfer_heat_loss_factor"]
        self.burnHeatLossFactor = mod_settings["burn_heat_loss_factor"]

        # Computed values
        self.cellTotalMass = self.fuelMass + self.waterMass + self.unburnableMass
        self.cellSpecificHeat = (
            self.fuelMass * self.fuelSpecificHeat +
            self.waterMass * self.waterSpecificHeat +
            self.unburnableMass * self.unburnableSpecificHeat
        ) / self.cellTotalMass

        self.cellThermalE = self.cellSpecificHeat * self.cellTemperature * self.cellTotalMass
        self.fuelThermalEnergy = self.fuelSpecificHeat * self.fuelTemperature * self.fuelMass
        self.waterThermalEnergy = self.waterSpecificHeat * self.waterTemperature * self.waterMass
        self.unburnableThermalEnergy = self.unburnableSpecificHeat * self.unburnableTemperature * self.unburnableMass

        self.fireIntensity = np.zeros((self.gridSizeY, self.gridSizeX))

        self.cellTemperature = np.where(initial_igniting_cells == 1, 1200, self.cellTemperature)
        self.updateCellsThermalEBasedOnTemp()

        self.neighbors_kernel, self.neighbors_vector_list_array = grid_and_simulation_tools.generate_kernel_and_vectors(self.cellEffectRadius, self.cellSizeX, self.cellSizeY)

        self.heightGradients = {}

        # TRANSFER HEAT --------------------------------
        for vector in self.neighbors_vector_list_array:
            # calculations for heat transfer from neighbor to cell
            ops_vector = -vector # note - sign: if neighbor is at (1,0), the transfer is (-1,0)

            dx = ops_vector[1] * self.cellSizeX
            dy = ops_vector[0] * self.cellSizeY

            real_vector = [dy, dx]
            real_vector_lenght = math.sqrt(dx*dx + dy*dy)
            normalized_real_vector = [n / real_vector_lenght for n in real_vector]

            self.heightGradients[f"{ops_vector}"] = grid_and_simulation_tools.calculate_delta_height_array_with_vector(self.cellHeight, ops_vector)

    # GRID SIMULATION
    def runSimStep(self, dt=1):
        self.specialModifications()
        self.loseHeat(dt)
        self.transferHeat(dt)
        self.updateTempsAndWaterContent()
        self.burn(dt)

    # GRID "STEP FUNCTIONS" FUNCTIONS
    def updateCellsThermalEBasedOnTemp(self):
        self.cellThermalE = self.cellSpecificHeat*self.cellTemperature*self.cellTotalMass # kJ / m**2
    
    def updateGlobalCellMasses(self):
        # Heal data
        self.fuelMass = np.clip(self.fuelMass, 0, None)
        self.waterMass = np.clip(self.waterMass, 0, None)
        self.unburnableMass = np.clip(self.unburnableMass, 0, None)


        self.cellTotalMass = self.fuelMass + self.waterMass + self.unburnableMass # kg / m**2
        self.cellSpecificHeat = (self.unburnableMass*self.unburnableSpecificHeat + self.fuelMass*self.fuelSpecificHeat + self.waterMass*self.waterSpecificHeat)/self.cellTotalMass # kJ / kg*K

    # GRID STEP FUNCTIONS
    def specialModifications(self):
        pass

    def transferHeat(self, dt):
        '''
        Funtion that manages all the heat transmission between cells. Simplified approach for the moment, possible
        application of conduction, convection and radiation in future.
        
        '''
        # copies to modify those properties and just change real value after all changes (for stability)
        newTE = np.copy(self.cellThermalE)

        # TRANSFER HEAT --------------------------------
        for vector in self.neighbors_vector_list_array:
            # calculations for heat transfer from neighbor to cell
            # vector preparation ----------------------
            ops_vector = -vector # note - sign: if neighbor is at (1,0), the transfer is (-1,0)

            dx = ops_vector[1] * self.cellSizeX
            dy = ops_vector[0] * self.cellSizeY

            real_vector = [dy, dx]
            real_vector_lenght = math.sqrt(dx*dx + dy*dy)
            normalized_real_vector = [n / real_vector_lenght for n in real_vector]
        
            # shift temps and deltaTemp ---------------
            shifted_temps = grid_and_simulation_tools.shift_array_and_fill_with_value(np.copy(self.cellTemperature), ops_vector, self.ambientTemp)
            deltaT = shifted_temps - self.cellTemperature

            # calculate conduction rate ---------------
            crossSectionLenght = math_and_geometry_tools.get_1d_cross_section_lenght(dx,dy,self.cellSizeX,self.cellSizeY)
            conductionRate = self.heatTransferRate * crossSectionLenght * deltaT / real_vector_lenght
            
            # wind modifier ---------------------------
            shifted_wind = grid_and_simulation_tools.shift_array_and_fill_with_value(np.copy(self.windField), ops_vector, np.copy(self.boundaryWindVector))
            shifted_wind_lenght = np.sqrt(shifted_wind[:,:,0]**2 + shifted_wind[:,:,1]**2)

            normalized_shifted_wind = np.stack((
                np.where(shifted_wind_lenght!=0, shifted_wind[:,:,0]/shifted_wind_lenght, 0),
                np.where(shifted_wind_lenght!=0, shifted_wind[:,:,1]/shifted_wind_lenght, 0)
            ), axis=-1)

            dotProduct = normalized_real_vector[0]*normalized_shifted_wind[:,:,0] + normalized_real_vector[1]*normalized_shifted_wind[:,:,1]
            windEffectCoef = np.exp(self.windEffectFactor * np.log1p(shifted_wind_lenght) * dotProduct)

            # height modifier -------------------------
            deltaHeight = self.heightGradients[f"{ops_vector}"]
            totalDistance = np.sqrt(deltaHeight**2 + real_vector_lenght**2)
            dHeightEffectCoef = np.exp(self.slopeEffectFactor * (deltaHeight/totalDistance))

            # apply modifiers -------------------------
            conductionRate *= windEffectCoef
            conductionRate *= dHeightEffectCoef

            # transfer heat ---------------------------
            Q = conductionRate * dt
            QperM2 = Q/self.cellArea
            newTE = np.where(QperM2 > 0, newTE + QperM2 * self.transferHeatLossFactor, newTE + QperM2)
        
        # update real values with temporary computation variables
        self.cellThermalE = np.copy(newTE)
        self.cellTemperature = self.cellThermalE/(self.cellTotalMass*self.cellSpecificHeat)

    def updateTempsAndWaterContent(self):
        self.fuelTemperature = np.copy(self.cellTemperature)
        self.unburnableTemperature = np.copy(self.cellTemperature)
        
        self.fuelThermalEnergy = self.fuelMass * self.fuelSpecificHeat * self.fuelTemperature
        self.unburnableThermalEnergy = self.unburnableMass * self.unburnableSpecificHeat * self.unburnableTemperature
        
        cellTemps = np.copy(self.cellTemperature)
        waterQs = self.cellThermalE - (self.fuelThermalEnergy + self.unburnableThermalEnergy) # get water thermal energy
        neededQsfor100C = self.waterMass*self.waterSpecificHeat*self.waterBoilingTemp # calculate wich thermal energy would water have at boiling temp
        
        waterEvaporates = np.where(cellTemps>self.waterBoilingTemp, True, False) # get where water evaporates (temp>boiling temp)
        massDeduction = np.where(waterEvaporates, (waterQs-neededQsfor100C)/self.waterLatentHeat, 0)
        massDeductionClamped = np.where(massDeduction>self.waterMass, self.waterMass, massDeduction)
        newWaterMasses = self.waterMass-massDeductionClamped # calculate new water mass accounting for mass loss (based on E)
        finalWaterMasses = np.maximum(newWaterMasses, 0.0) # clamp: if it goes below 0, it stays at 0
        
        finalWaterTemp = np.clip(cellTemps, None, self.waterBoilingTemp) # clamp: if > 100, stays at 100 (over that translated to evaporation)
        
        self.waterTemperature = np.copy(finalWaterTemp)
        self.waterMass = finalWaterMasses
        self.waterThermalEnergy = self.waterMass * self.waterSpecificHeat * self.waterTemperature
        
        # Recalculate cell thermal e after mass loss. Temp remains same
        self.updateGlobalCellMasses()
        self.updateCellsThermalEBasedOnTemp()

    def loseHeat(self, dt):
        # The temperature decay factor towards ambient temperature
        self.cellTemperature = self.initialConditions['cellTemperature'] + (self.cellTemperature - self.initialConditions['cellTemperature']) * (self.heatLossFactor ** dt)
        
        # Update thermal behavior based on the new temperature
        self.updateCellsThermalEBasedOnTemp()

    def burn(self, dt):
        self.updateCellsThermalEBasedOnTemp()
        burning = np.where(self.fuelTemperature > self.fuelIgnitingTemp, 1, 0) # 1 where temp allows burning
        self.fireIntensity = burning * self.fuelMass * self.fuelBurnRate * self.fuelCalorificValue # calculate fire intensity --> kJ/m^2*s
    
        # burn also based on neighbors
        neighborBurnFactor = 0.05
        neighbors_kernel = np.array([[0,neighborBurnFactor,0],[neighborBurnFactor,1-neighborBurnFactor*4,neighborBurnFactor],[0,neighborBurnFactor,0]])
        self.fireIntensity = convolve2d(self.fireIntensity, neighbors_kernel, mode='same', fillvalue=0)
    
        # calculate released energy accounting for neighbors
        idealReleasedEnergy = self.fireIntensity * dt # per square meter
        usedEnergy = np.clip(idealReleasedEnergy, None, self.fuelMass*self.fuelCalorificValue)
        realReleasedEnergy = usedEnergy * self.burnHeatLossFactor
    
    
        self.fuelMass -= (usedEnergy/self.fuelCalorificValue)
        self.fuelMass = np.clip(self.fuelMass, 0, None)
        self.updateGlobalCellMasses()
        self.cellTemperature = (self.cellThermalE+realReleasedEnergy)/(self.cellTotalMass*self.cellSpecificHeat) # update cell temp based on E released
        self.updateCellsThermalEBasedOnTemp()

class Toolbox():
    @staticmethod
    def celsiusToKelvin(t):
        return t+273.15
    
    @staticmethod
    def kelvinToCelsius(t):
        return t-273.15

class FramesRecorder():
    def __init__(self, gridHolder):
        self.data = {
            "timestamps": [],
            "fuel_moisture_percentage": {"data": [], 'colormap': 'blues', 'min': float('inf'), 'max': float('-inf')},
            "temperature_celsius": {"data": [], 'colormap': 'hot', 'min': float('inf'), 'max': float('-inf')},
            "fire_intensity_kW_m2": {"data": [], 'colormap': 'viridis', 'min': float('inf'), 'max': float('-inf')},
            "fuel_mass_kg": {"data": [], 'colormap': 'turbid_r', 'min': float('inf'), 'max': float('-inf')},
            "fire_time_evolution": {"data": [], 'colormap': 'turbo', 'min': float('inf'), 'max': float('-inf')}
        }

        self.simulationProgress = 0
        self.referenceGrid = gridHolder

    def reset(self):
        self.__init__(self.referenceGrid)

    def _update_min_max(self, key, array):
        arr_min = np.nanmin(array)
        arr_max = np.nanmax(array)
        self.data[key]['min'] = min(self.data[key]['min'], arr_min)
        self.data[key]['max'] = max(self.data[key]['max'], arr_max)

    def record(self, second, totalTime):
        self.data["timestamps"].append(second)

        # Fuel moisture (%)
        moisture = (np.where(self.referenceGrid.fuelMass == 0, 0, self.referenceGrid.waterMass / self.referenceGrid.fuelMass)) * 100
        self.data['fuel_moisture_percentage']["data"].append(np.copy(moisture))
        self._update_min_max('fuel_moisture_percentage', moisture)

        # Temperature (Celsius)
        temp_c = self.referenceGrid.cellTemperature - 273.15
        self.data['temperature_celsius']["data"].append(np.copy(temp_c))
        self._update_min_max('temperature_celsius', temp_c)

        # Fire intensity (kW/m2)
        intensity = self.referenceGrid.fireIntensity
        self.data['fire_intensity_kW_m2']["data"].append(np.copy(intensity))
        self._update_min_max('fire_intensity_kW_m2', intensity)

        # Fuel mass (kg)
        fuel_mass = self.referenceGrid.fuelMass
        self.data["fuel_mass_kg"]["data"].append(np.copy(fuel_mass))
        self._update_min_max('fuel_mass_kg', fuel_mass)

        # Fire time evolution
        # Initialize fire_time_evolution for the first record if it's empty
        if not self.data["fire_time_evolution"]["data"]:
            # All cells are initially not burned (or set to initial state for comparison)
            initial_fire_state = np.zeros_like(fuel_mass, dtype=float) # Or np.full_like(fuel_mass, np.nan)
            self.data["fire_time_evolution"]["data"].append(initial_fire_state)

        # Get the previous fuel mass and fire time evolution states
        if len(self.data["fuel_mass_kg"]["data"]) > 1:
            previous_fuel_mass = self.data["fuel_mass_kg"]["data"][-2]
        else:
            previous_fuel_mass = self.data["fuel_mass_kg"]["data"][-1]
        previous_fire_time_evolution = self.data["fire_time_evolution"]["data"][-1] # This is already the last appended element

        # Determine where fuel mass has decreased (indicating burning) AND where it was previously not marked as burned
        # You might want a threshold for "significantly decreased" rather than just any decrease
        # For simplicity, let's assume a decrease means it's burning.
        # This condition will be true for cells that are currently burning AND have reduced fuel.
        burning_condition = np.less(fuel_mass, previous_fuel_mass)

        # Update fire_time_evolution:
        # If a cell is burning (burning_condition is True) AND it hasn't been marked as burned yet (previous_fire_time_evolution == 0),
        # then set its time to the current 'second'. Otherwise, keep its previous time.
        current_fire_time_evolution = np.where(
            np.logical_and(burning_condition, previous_fire_time_evolution == 0),
            second/60, #minutes
            previous_fire_time_evolution
        )
        self.data["fire_time_evolution"]["data"].append(np.copy(current_fire_time_evolution))
        self._update_min_max('fire_time_evolution', current_fire_time_evolution)
    
def simulate(gridHolder, recorderHolder, deltaTime, totalTime, frameRecordInterval):
    print("")
    print('UTS-- PREPARING SIMULATION')
    cache.set("progress", recorderHolder.simulationProgress)
    cache.set("last_frame_sent", 0)
    cache.set("new_frames", 
            {   "timestamps": [],
                "fuel_moisture_percentage": {"data": [], 'colormap': None, 'min': None, 'max': None},
                "temperature_celsius": {"data": [], 'colormap': None, 'min': None, 'max': None},
                "fire_intensity_kW_m2": {"data": [], 'colormap': None, 'min': None, 'max': None},
                "fuel_mass_kg": {"data": [], 'colormap': None, 'min': None, 'max': None},
                "fire_time_evolution": {"data": [], 'colormap': None, 'min': None, 'max': None},
            })
    cache.set("all_frames", 
            {   "timestamps": [],
                "fuel_moisture_percentage": {"data": [], 'colormap': None, 'min': None, 'max': None},
                "temperature_celsius": {"data": [], 'colormap': None, 'min': None, 'max': None},
                "fire_intensity_kW_m2": {"data": [], 'colormap': None, 'min': None, 'max': None},
                "fuel_mass_kg": {"data": [], 'colormap': None, 'min': None, 'max': None},
                "fire_time_evolution": {"data": [], 'colormap': None, 'min': None, 'max': None}
            })
    test_frames = {"timestamps": [],
                        "fuel_moisture_percentage": {"data": [], 'colormap': None, 'min': None, 'max': None},
                        "temperature_celsius": {"data": [], 'colormap': None, 'min': None, 'max': None},
                        "fire_intensity_kW_m2": {"data": [], 'colormap': None, 'min': None, 'max': None},
                        "fuel_mass_kg": {"data": [], 'colormap': None, 'min': None, 'max': None},
                        "fire_time_evolution": {"data": [], 'colormap': None, 'min': None, 'max': None}
                    }
    
    print("    |- Setting cache initial state")
    print(f"      |- Cache progress: \033[92m OK \033[0m") if cache.get("progress") == 0 else print(f" |- Cache progress: \033[93m ERROR \033[0m")
    print(f"      |- Cache last frame sent: \033[92m OK \033[0m") if cache.get('last_frame_sent') == 0 else print(f" |- Cache last frame sent: \033[93m ERROR \033[0m")
    print(f"      |- Frames cache:")
    print(f"         |- Cache new frames: \033[92m OK \033[0m") if cache.get('new_frames') == test_frames else print(f"    |- Cache new frames: \033[93m ERROR \033[0m")
    print(f"         |- Cache all frames: \033[92m OK \033[0m") if cache.get('all_frames') == test_frames else print(f"    |- Cache all frames: \033[93m ERROR \033[0m")
    print("")
    print("UTS-- SIMULATION STARTED")
    print("\033[93mtwo 'invalid value encountered in divide' errors will show up. it's normal \033[0m")
    print("")

    elapsedTime = 0  # initialize a variable to keep track of time
    elapsedTimeForPlotRecord = 0

    gridHolder.runSimStep(1)
    elapsedTime += 1
    elapsedTimeForPlotRecord += 1
    recorderHolder.record(1, totalTime)
    
    while elapsedTime < totalTime:
        gridHolder.runSimStep(deltaTime)
        elapsedTime += deltaTime
        recorderHolder.simulationProgress = (elapsedTime / totalTime) * 100
        cache.set("progress", recorderHolder.simulationProgress)
        elapsedTimeForPlotRecord += deltaTime
        
        if elapsedTimeForPlotRecord >= (frameRecordInterval):  # Check if 30 minutes (1800 seconds) have passed
            recorderHolder.record(elapsedTime, totalTime)
            elapsedTimeForPlotRecord = 0  # reset the counter after recording
            
            dataToSend = cache.get("new_frames")

            for frame in range(int(cache.get("last_frame_sent")), len(recorderHolder.data["timestamps"])):
                dataToSend["timestamps"].append(recorderHolder.data["timestamps"][frame])
                for key in ["fuel_moisture_percentage", "temperature_celsius", "fire_intensity_kW_m2", "fuel_mass_kg", "fire_time_evolution"]:
                    dataToSend[key]["data"].append(recorderHolder.data[key]["data"][frame])

            for key in ["fuel_moisture_percentage", "temperature_celsius", "fire_intensity_kW_m2", "fuel_mass_kg", "fire_time_evolution"]:
                dataToSend[key]["min"] = recorderHolder.data[key]["min"]
                dataToSend[key]["max"] = recorderHolder.data[key]["max"]
                dataToSend[key]["colormap"] = recorderHolder.data[key]["colormap"]

            print(f"UTS-- PLOT UPDATE (progress: {round(cache.get('progress'), 2):05.2f} %) --> timestamps to send: {dataToSend['timestamps']}")

            cache.set("last_frame_sent", len(recorderHolder.data["timestamps"]))
            cache.set("new_frames", dataToSend)
            cache.set("all_frames", recorderHolder.data)
    
    cache.set("progress", 100)
    
    print("")
    print(f"UTS-- PLOT UPDATE --> last update with all timestamps: {cache.get('all_frames')['timestamps']}")
    print(f"UTS-- SIMULATION FINISHED")
        