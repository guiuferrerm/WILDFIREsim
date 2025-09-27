# WILDFIREsim

_A Catalan baccalaureate research project (Treball de Recerca, TDR) focused on the simulation of wildfires._

This application is designed to simulate the behavior of wildfires using a simplified, grid-based model. It incorporates real-world topographical data and provides an interactive interface for defining, visualizing, and reviewing wildfire scenarios.

Geographical elevation data used in this project has been primarily obtained from:  
[https://viewfinderpanoramas.org/dem3.html](https://viewfinderpanoramas.org/dem3.html)

---

## Application Overview

The application is structured into the following pages:

- **Inici (Home)**  
  Provides a general introduction to the simulator and its objectives.

- **Simulació (Simulation)**  
  Runs the actual wildfire simulation based on previously defined conditions. It displays the fire’s progression on a grid representing the terrain.

- **Creació de fitxers (File Creation)**  
  Allows users to define the initial conditions of the fire (such as wind or fuel type). These conditions are then saved in a `.wfss` (WILDFIREsim Settings) file format.

- **Revisió de fitxers (File Review)**  
  Enables users to quickly review the contents of existing `.wfss` files to verify settings before simulation.

---

## How to Run the Simulator

1. Make sure you have Python installed.

2. Install the required Python libraries. You can use `pip`:

   ```bash
   pip install numpy scipy plotly dash diskcache
   ```

3. Clone or download the repository.

4. Run the main file:

   ```bash
   python index.py
   ```

5. A browser window should open automatically. If it does not, copy the URL shown in the terminal (usually `http://127.0.0.1:8050`) and open it manually in your web browser.

---

## Dependencies

- Python 3.x
- numpy
- scipy
- plotly
- dash
- diskcache
- math (standard library)
- base64 (standard library)
- io (standard library)
- os (standard library)

---

## File Format

`.wfss` files are npz configuration files used by the simulator to store wildfire scenario parameters. These include:

- Wind x and y components
- Fuel distribution
- Terrain elevation
- Fuel moisture
- Others

---

## Author

This project was developed as part of a **Treball de Recerca (TDR)** in Catalonia, focusing on modeling wildfire dynamics with simple tools and real-world data.

