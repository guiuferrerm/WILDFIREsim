import numpy as np
import math
from scipy.signal import convolve2d

class SimGrid:
    # GRID SETUP, CREATION AND RESET
    def __init__(self):
        # Geometry
        self.cellSize = 0
        self.cellArea = 0
        self.gridSizeX = 0
        self.gridSizeY = 0
        self.Xmesh = None
        self.Ymesh = None
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
        self.boundaryWindVector = None  # Will be np.array([x, y])

        # Wind
        self.windField = None  # 3D array (Y, X, 2)

        # Fire Spread Constants
        self.heatTransferRate = 0
        self.slopeEffectFactor = 0
        self.windAffectCt = 0
        self.heatLossFactor = 0

        # Radiation
        self.stefanBoltzmannCt = 0

    def reset(self):
        self.__init__()

    def setup(self,
            heightArray, XmeshArray, YmeshArray, temperatureArray, fuelMoistureArray, fuelMassArray, unburnableMassArray, windXArray, windYArray,
            unmod_settings, mod_settings, metadata,
            initial_igniting_cells):

        # Grid structure
        self.cellSize = float(unmod_settings["cell_size"])
        self.cellArea = self.cellSize ** 2
        self.gridSizeX = int(unmod_settings["array_dim_x"])
        self.gridSizeY = int(unmod_settings["array_dim_y"])

        self.Xmesh = np.copy(XmeshArray)
        self.Ymesh = np.copy(YmeshArray)
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
        self.stefanBoltzmannCt = unmod_settings["stefan_boltzmann_ct"]

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

        self.heatTransferRate = mod_settings["heat_transfer_rate"]
        self.slopeEffectFactor = mod_settings["slope_effect_factor"]
        self.windAffectCt = mod_settings["wind_effect_constant"]
        self.fuelBurnRate = mod_settings["fuel_burn_rate"]
        self.heatLossFactor = mod_settings["heat_loss_factor"]

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
        self.cellTotalMass = self.fuelMass + self.waterMass + self.unburnableMass # kg / m**2
        self.cellSpecificHeat = (self.unburnableMass*self.unburnableSpecificHeat + self.fuelMass*self.fuelSpecificHeat + self.waterMass*self.waterSpecificHeat)/self.cellTotalMass # kJ / kg*K

    # GRID STEP FUNCTIONS
    def specialModifications(self):
        pass

    def transferHeat(self, dT):
        '''
        Funtion that manages all the heat transmission between cells. Accounting for a simplified approach for the moment, possible
        application of conduction, convection and radiation in future.
        
        '''
        # copies to modify those properties and just change real value after all changes (for stability)
        newTE = np.copy(self.cellThermalE)
        
        # CALCULATE STABLE TEMP -----------------------
        # cells considered neighbors and with wich we calculate heat transfer
        neighbors_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
        
        qArray = self.cellTotalMass*self.cellSpecificHeat*self.cellTemperature
        qDivArray = self.cellTotalMass*self.cellSpecificHeat
        
        boundaryQ = self.boundaryMass*self.boundaryCe*self.ambientTemp
        boundaryDiv =self.boundaryMass*self.boundaryCe
        
        convoluted_qArray = convolve2d(qArray, neighbors_kernel, mode='same', fillvalue=boundaryQ)
        convoluted_DivArray = convolve2d(qDivArray, neighbors_kernel, mode='same', fillvalue=boundaryDiv)
        
        stableTempArray = convoluted_qArray/convoluted_DivArray
        # ---------------------------------------------
        
        neighbors_vectors = np.array([[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1]]) # From origin to surroundings
        
        for vector in neighbors_vectors:
            # calculations for heat transfer from neighbor to cell
            ops_vector = -vector # note - sign: if neighbor is at (1,0), the transfer is (-1,0)
            ops_vector_lenght = math.sqrt(ops_vector[0]**2 + ops_vector[1]**2)
            normalized_ops_vector = ops_vector/ops_vector_lenght
        
            # shift temps and deltaTemp
            temp = np.copy(stableTempArray)
            boundary_value = self.ambientTemp
            
            if ops_vector[0] > 0:
                temp[-ops_vector[0]:, :] = boundary_value
            elif ops_vector[0] < 0:
                temp[:-ops_vector[0], :] = boundary_value
            if ops_vector[1] > 0:
                temp[:, -ops_vector[1]:] = boundary_value
            elif ops_vector[1] < 0:
                temp[:, :-ops_vector[1]] = boundary_value
            
            shifted_stable_temp = np.roll(temp, (ops_vector[0], ops_vector[1]), axis=(0,1))
            deltaTemps = shifted_stable_temp - self.cellTemperature
            
            # Q transfers
            idealQTransfers = self.heatTransferRate*self.cellTotalMass*self.cellSpecificHeat*deltaTemps
            
            # shift wind
            wind = np.copy(self.windField)
            boundary_value = np.copy(self.boundaryWindVector)
            
            if ops_vector[0] > 0:
                wind[-ops_vector[0]:, :] = boundary_value
            elif ops_vector[0] < 0:
                wind[:-ops_vector[0], :] = boundary_value
            if ops_vector[1] > 0:
                wind[:, -ops_vector[1]:] = boundary_value
            elif ops_vector[1] < 0:
                wind[:, :-ops_vector[1]] = boundary_value
            
            shifted_wind = np.roll(wind, (ops_vector[0], ops_vector[1]), axis=(0, 1))
            shifted_wind_lenght = np.sqrt(shifted_wind[:,:,0]**2 + shifted_wind[:,:,1]**2)
            
            # shift height
            height = np.copy(self.cellHeight)
            
            if ops_vector[0] > 0:
                height[-ops_vector[0]:, :] = height[0,:].reshape(1, -1)
            elif ops_vector[0] < 0:
                height[:-ops_vector[0], :] = height[-1,:].reshape(1, -1)
            if ops_vector[1] > 0:
                height[:, -ops_vector[1]:] = height[:,0].reshape(-1, 1)
            elif ops_vector[1] < 0:
                height[:, :-ops_vector[1]] = height[:,-1].reshape(-1, 1)
            
            shifted_height = np.roll(height, (ops_vector[0], ops_vector[1]), axis=(0,1))
            deltaHeight = self.cellHeight - shifted_height
            
            # calculations
            # shifted_wind_lenght = np.nan_to_num(shifted_wind_lenght, nan=0, posinf=0, neginf=0)
            
            normalized_shifted_wind = np.stack((
                np.where(shifted_wind_lenght!=0, shifted_wind[:,:,0]/shifted_wind_lenght, 0),
                np.where(shifted_wind_lenght!=0, shifted_wind[:,:,1]/shifted_wind_lenght, 0)
            ), axis=-1)
            
            dotProduct = normalized_ops_vector[0]*normalized_shifted_wind[:,:,0] + normalized_ops_vector[1]*normalized_shifted_wind[:,:,1]
            windEffectCoef = np.exp(self.windAffectCt * (shifted_wind_lenght / (self.cellSize*ops_vector_lenght)) * dotProduct)
            
            qTransfer = idealQTransfers * windEffectCoef + idealQTransfers * windEffectCoef * (deltaHeight/(np.sqrt(deltaHeight**2 + ops_vector_lenght**2))) * self.slopeEffectFactor
            
            newTE += qTransfer * dT
        
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
        newWaterMasses = np.where(waterEvaporates, self.waterMass-((waterQs-neededQsfor100C)/self.waterLatentHeat), self.waterMass) # calculate new water mass accounting for mass loss (based on E)
        finalWaterMasses = np.where(newWaterMasses >= 0, newWaterMasses, 0) # clamp: if it goes below 0, it stays at 0
        
        finalWaterTemp = np.where(cellTemps <= 100, cellTemps, 100) # clamp: if > 100, stays at 100 (over that translated to evaporation)
        
        self.waterTemperature = np.copy(finalWaterTemp)
        self.waterMass = finalWaterMasses
        self.waterThermalEnergy = self.waterMass * self.waterSpecificHeat * self.waterTemperature
        
        # Recalculate cell thermal e after mass loss. Temp remains same
        self.updateGlobalCellMasses()
        self.updateCellsThermalEBasedOnTemp()

    def loseHeat(self, dT):
        # The temperature decay factor towards ambient temperature
        self.cellTemperature = self.initialConditions['cellTemperature'] + (self.cellTemperature - self.initialConditions['cellTemperature']) * (self.heatLossFactor ** dT)
        
        # Update thermal behavior based on the new temperature
        self.updateCellsThermalEBasedOnTemp()

    def burn(self, dT):
        burning = np.where(self.fuelTemperature > self.fuelIgnitingTemp, 1, 0) # 1 where temp allows burning
        self.fireIntensity = burning * self.fuelBurnRate * self.fuelMass * self.fuelCalorificValue # calculate fire intensity --> kJ/m^2*s
    
        # burn also based on neighbors
        neighborBurnFactor = 0.1
        neighbors_kernel = np.array([[0,neighborBurnFactor,0],[neighborBurnFactor,1,neighborBurnFactor],[0,neighborBurnFactor,0]])
        self.fireIntensity = convolve2d(self.fireIntensity, neighbors_kernel, mode='same', fillvalue=0)
    
        # calculate released energy accounting for neighbors
        releasedEnergy = self.fireIntensity * dT
    
    
        self.fuelMass -= (releasedEnergy/self.fuelCalorificValue)
        self.updateGlobalCellMasses()
        self.cellTemperature = (self.cellThermalE+releasedEnergy)/(self.cellTotalMass*self.cellSpecificHeat) # update cell temp based on E released
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
            "fuel_mass_kg": {"data": [], 'colormap': 'turbid_r', 'min': float('inf'), 'max': float('-inf')}
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
        moisture = (self.referenceGrid.waterMass / self.referenceGrid.fuelMass) * 100
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

        # Progress
        self.simulationProgress = (second * 100) / totalTime
    
def simulate(gridHolder, recorderHolder, deltaTime, totalTime, frameRecordInterval):
    elapsedTime = 0  # initialize a variable to keep track of time
    elapsedTimeForPlotRecord = 0

    gridHolder.runSimStep(1)
    elapsedTime += 1
    elapsedTimeForPlotRecord += 1
    recorderHolder.record(1, totalTime)
    
    while elapsedTime < totalTime:
        gridHolder.runSimStep(deltaTime)
        elapsedTime += deltaTime
        elapsedTimeForPlotRecord += deltaTime
        
        if elapsedTimeForPlotRecord >= (frameRecordInterval):  # Check if 30 minutes (1800 seconds) have passed
            recorderHolder.record(elapsedTime, totalTime)
            elapsedTimeForPlotRecord = 0  # reset the counter after recording
    
    print("finished simulation")
        