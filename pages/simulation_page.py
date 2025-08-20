import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx, MATCH, ALL
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app import app
from utils.cache_config import background_callback_manager, cache
from utils import cache_config
from utils.hgt_file_management import HGT_to_np_array, prepare_HGT_as_array_data
from utils.npz_file_management import read_and_store_npz_contents
from utils.dcc_upload_management import read_and_store_dcc_file_at
from utils import fire_simulation
from utils import time_conversion_tools

layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                dcc.Upload(
                    id='upload-wfss-for-simulation',
                    className='file-upload',
                    children=html.Button('Upload .wfss file', className="button"),
                    multiple=False,  # Allow only one file for simplicity
                    accept='.wfss',
                ),
                html.Div(id='output-data-upload-simulation', className="error-message"),
                html.Div("No file uploaded", id='upload-status-simulation', className="error-message"),
            ], className="box-section"),
            # Store the uploaded file data
            dcc.Store(id='uploaded-file-store-simulation'),
            dcc.Store(id='figure-store'),
            dcc.Store(id='settings-store-simulation'),
            dcc.Store(id='simulation-setup-settings-store', data={'time_step': 3, 'total_time': 3600}), # must be equal to initial input value
            dcc.Store(id='store-igniting-cells'),
            dcc.Store(id='simulation-status-store', data={'state': 'preparing', 'progress': 0}),
            dcc.Store(id='sim-frame-store', data={
                "timestamps": [],
                "fuel_moisture_percentage": {"data": [], 'colormap': 'blues', 'min': float('inf'), 'max': float('-inf')},
                "temperature_celsius": {"data": [], 'colormap': 'hot', 'min': float('inf'), 'max': float('-inf')},
                "fire_intensity_kW_m2": {"data": [], 'colormap': 'viridis', 'min': float('inf'), 'max': float('-inf')},
                "fuel_mass_kg": {"data": [], 'colormap': 'turbid_r', 'min': float('inf'), 'max': float('-inf')},
                "fire_time_evolution": {"data": [], 'colormap': 'turbo', 'min': float('inf'), 'max': float('-inf')}
            }),  # Holds the simulation frames
            dcc.Store(id='is-first-plot-store', data=True),
            dcc.Interval(
                id='frame-interval',
                interval=2000,  # in milliseconds
                n_intervals=0,  # starting at 0 intervals
                disabled=True  # initially disabled, enabled after simulation starts
            ),
            html.Div([
                html.Div([
                    html.H2("Settings"),
                    html.H4("version", id="version-label-simulation"),
                    html.Div([
                        html.H3("Simulation-Specific Settings"),
                        html.Div([
                            html.Label("time_step"),
                            dcc.Input(id="simulation-time-step-input", value=3, type='number'),
                        ], className="input-row"),
                        html.Div([
                            html.Label("total_time"),
                            dcc.Input(id="simulation-total-time-input", value=3600, type='number'),
                        ], className="input-row"),
                        html.Div([
                            html.Label("record_interval"),
                            dcc.Input(id="simulation-record-interval-input", value=900, type='number'),
                        ], className="input-row"),
                    ], id='static-inputs-simulation'),
                    html.Div([
                        
                    ], id='dynamic-inputs-simulation'),
                ], id='all-inputs-simulation', className='all-inputs-simulation'),
            ], className="box-section"),
        ], className='simulation-input'),
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Dropdown(
                            id='data-type-dropdown-simulation',
                            options=[
                                {'label': 'temperature (C°)', 'value': 'temperature_celsius'},
                                {'label': 'fire intensity (kW/m2)', 'value': 'fire_intensity_kW_m2'},
                                {'label': 'fuel mass (kg/m2)', 'value': 'fuel_mass_kg'},
                                {'label': 'fuel moisture (% mass)', 'value': 'fuel_moisture_percentage'},
                                {'label': 'fire evolution (minutes)', 'value': 'fire_time_evolution'},
                            ],
                            value='temperature_celsius',  # Default value
                            style={'width': '200px'},
                            clearable=False,  # No "reset" button
                            searchable=False
                        ),
                        dcc.Dropdown(
                            id='meshgrid-type-dropdown-simulation',
                            options=[
                                {'label': 'lat/lon degrees', 'value': 'meshgridlatlondeg'},
                                {'label': 'meters', 'value': 'meshgridmeters'},
                            ],
                            value='meshgridlatlondeg',  # Default value
                            style={'width': '200px'},
                            clearable=False,  # No "reset" button
                            searchable=False
                        ),
                    ], className='box-section'),
                    dcc.Graph(id='simulation-plot'),
                    html.Div([
                        html.H3('time passed', id='simulation-frame-time'),
                        html.H3('progress: ', id='simulation-progress'),
                    ], className='box-section'),
                    # Simulation Controls (below the plot)
                    html.Div([
                        html.Button("Simulate", id="simulate-btn", className="button", n_clicks=0, disabled=False),
                        html.Button("Reset", id="reset-btn", className="button", n_clicks=0, disabled=True),
                        html.Br(),
                        dcc.Slider(
                            id="frame-slider",
                            min=0,
                            max=100,  # Example max frames
                            step=1,
                            marks={i: f"Frame {i}" for i in range(0, 101, 10)},
                            value=0,
                            disabled=True,  # Initially disabled until simulate is pressed
                        ),
                    ], className="box-section"),
                ], className="simulation-controls")
            ], className="box-section"),
        ], className='simulation-output'),
    ], className='simulation-wrap'),
])

simulation_grid = fire_simulation.SimGrid()
frames_recorder = fire_simulation.FramesRecorder(simulation_grid)
disableIntervalNextCall = False


# Upload
@app.callback(
    output=[
        Output('output-data-upload-simulation', 'children'),
        Output('upload-status-simulation', 'children'),
        Output('uploaded-file-store-simulation', 'data'),
        Output('output-data-upload-simulation', 'className'),
        Output('upload-status-simulation', 'className'),
    ],
    inputs=[Input('upload-wfss-for-simulation', 'contents')],
    state=[State('upload-wfss-for-simulation', 'filename')],
    prevent_initial_call=True
)
def handle_file_upload(contents, filename):
    if not contents or not filename:
        return 'No file uploaded', '', None, 'error-message', 'error-message'

    file_path = f"/tmp/{filename}"
    read_and_store_dcc_file_at(contents, filename, file_path)
    data_array = {}
    data_array = read_and_store_npz_contents(file_path)

    return (
        "File uploaded successfully!",
        f"Uploaded File: {filename}",
        data_array,
        "successful-message",
        "successful-message"
    )

@app.callback(
    output=[
        Output('version-label-simulation', 'children'),
        Output('dynamic-inputs-simulation', 'children'),
    ],
    inputs=[Input('uploaded-file-store-simulation', 'data')]
)
def update_dynamic_inputs(data_array):
    if not data_array:
        raise dash.exceptions.PreventUpdate

    version = float(data_array["metadata"]["version"])
    title = data_array["metadata"]['setup_title']
    mod_settings = data_array["mod_settings"]

    # Define your dynamic inputs here
    input_elements = [
        html.H3("Physical Properties"),
    ]

    for key, value in mod_settings.items():
        input_elements.append(
            html.Div([
                html.Label(key),
                dcc.Input(
                    id={'type': 'setting-input', 'key': key},  # Important: use pattern-matching ID
                    value=value,
                    type='number',
                    min=0
                ),
            ], className='input-row')
        )

    return f"version {version} | {title}", input_elements

@app.callback(
    output=[
        Output('simulation-plot', 'figure'),
        Output('store-igniting-cells', 'data'),
        Output('simulation-frame-time', 'children'),
        Output('simulation-progress', 'children'),
        Output('is-first-plot-store', 'data'),
        Output('figure-store', 'data'),
    ],
    inputs=[
        Input('simulate-btn', 'n_clicks'),
        Input('sim-frame-store', 'data'),
        Input('simulation-plot', 'relayoutData'),
        Input('frame-slider', 'value'),
        Input('uploaded-file-store-simulation', 'data'),
        Input('data-type-dropdown-simulation', 'value'),
        Input('meshgrid-type-dropdown-simulation', 'value'),
    ],
    state=[
        State('store-igniting-cells', 'data'),
        State('simulation-status-store', 'data'),
        State('is-first-plot-store', 'data'),
        State('figure-store', 'data'),
    ],
    prevent_initial_call=True
)
def update_plot_based_on_state(trigger_n_clicks, sim_frames, relayout_data, selected_frame, uploaded_data, data_shown, meshgrid_type, 
                               stored_ignition, simulation_status, is_first_plot, fig_store):

    global igniting_cells, X, Y

    trigger = ctx.triggered_id

    fig = go.Figure(fig_store)

    if uploaded_data is None:
        raise dash.exceptions.PreventUpdate

    fig_updated = False

    if simulation_status['state'] == 'running':
        if sim_frames:
            if 0 <= selected_frame < len(sim_frames[data_shown]["data"]):
                frame_data = np.array(sim_frames[data_shown]["data"][-1])
            else:
                #print(f"selected_frame {selected_frame} is out of range!")
                raise dash.exceptions.PreventUpdate
        else:
            #print("No data!!!")
            raise dash.exceptions.PreventUpdate
        
        if sim_frames and data_shown in sim_frames:
            frame_data = np.array(sim_frames[data_shown]["data"][-1])
            fig.data[0].x = np.array(uploaded_data["x_deg_mesh"])[0] if meshgrid_type == 'meshgridlatlondeg' else np.array(uploaded_data["x_meter_mesh"])[0]
            fig.data[0].y = np.array(uploaded_data["y_deg_mesh"])[:, 0] if meshgrid_type == 'meshgridlatlondeg' else np.array(uploaded_data["y_meter_mesh"])[:, 0]
            fig.data[2].x = np.array(uploaded_data["x_deg_mesh"])[0] if meshgrid_type == 'meshgridlatlondeg' else np.array(uploaded_data["x_meter_mesh"])[0]
            fig.data[2].y = np.array(uploaded_data["y_deg_mesh"])[:, 0] if meshgrid_type == 'meshgridlatlondeg' else np.array(uploaded_data["y_meter_mesh"])[:, 0]
            fig.data[2].z = frame_data
            fig.data[2].zmin = sim_frames[data_shown]["min"]
            fig.data[2].zmax = sim_frames[data_shown]["max"]
            fig.data[2].colorscale = sim_frames[data_shown]["colormap"]
            fig_updated = True
        else:
            raise dash.exceptions.PreventUpdate

        time_passed_in_seconds = sim_frames["timestamps"][-1]
        days_p, hours_p, minutes_p, seconds_p = time_conversion_tools.convert_seconds_to_dhms(time_passed_in_seconds)

        if fig_updated:
            fig.update_layout(
                    xaxis=dict(scaleanchor="y"),  # Link the x-axis to the y-axis
                    yaxis=dict(scaleanchor="x"),  # Link the y-axis to the x-axis
                    xaxis2=dict(matches='x1'),
                    yaxis2=dict(matches='y1'),
                    dragmode='zoom'  # Enable selection on the second plot
                )

        return (
            fig if fig_updated else dash.no_update,
            stored_ignition,
            f"time passed: {days_p}d {hours_p}h {minutes_p}m {seconds_p}s",
            f'progress: {round(simulation_status["progress"])} %, {simulation_status["state"]}',
            is_first_plot,
            fig if fig_updated else dash.no_update,
        )
    
    elif simulation_status['state'] == 'finished':
        if sim_frames:
            if 0 <= selected_frame < len(sim_frames[data_shown]["data"]):
                frame_data = np.array(sim_frames[data_shown]["data"][selected_frame])
            else:
                #print(f"selected_frame {selected_frame} is out of range!")
                raise dash.exceptions.PreventUpdate
        else:
            #print("No data!!!")
            raise dash.exceptions.PreventUpdate
        
        if sim_frames and data_shown in sim_frames:
            frame_data = np.array(sim_frames[data_shown]["data"][selected_frame])
            fig.data[0].x = np.array(uploaded_data["x_deg_mesh"])[0] if meshgrid_type == 'meshgridlatlondeg' else np.array(uploaded_data["x_meter_mesh"])[0]
            fig.data[0].y = np.array(uploaded_data["y_deg_mesh"])[:, 0] if meshgrid_type == 'meshgridlatlondeg' else np.array(uploaded_data["y_meter_mesh"])[:, 0]
            fig.data[2].x = np.array(uploaded_data["x_deg_mesh"])[0] if meshgrid_type == 'meshgridlatlondeg' else np.array(uploaded_data["x_meter_mesh"])[0]
            fig.data[2].y = np.array(uploaded_data["y_deg_mesh"])[:, 0] if meshgrid_type == 'meshgridlatlondeg' else np.array(uploaded_data["y_meter_mesh"])[:, 0]
            fig.data[2].z = frame_data
            fig.data[2].zmin = sim_frames[data_shown]["min"]
            fig.data[2].zmax = sim_frames[data_shown]["max"]
            fig.data[2].colorscale = sim_frames[data_shown]["colormap"]
            fig_updated = True
        else:
            raise dash.exceptions.PreventUpdate

        time_passed_in_seconds = sim_frames["timestamps"][selected_frame]
        days_p, hours_p, minutes_p, seconds_p = time_conversion_tools.convert_seconds_to_dhms(time_passed_in_seconds)

        if fig_updated:
            fig.update_layout(
                    xaxis=dict(scaleanchor="y"),  # Link the x-axis to the y-axis
                    yaxis=dict(scaleanchor="x"),  # Link the y-axis to the x-axis
                    xaxis2=dict(matches='x1'),
                    yaxis2=dict(matches='y1'),
                    dragmode='zoom'  # Enable selection on the second plot
                )

        return (
            fig if fig_updated else dash.no_update,
            stored_ignition,
            f"time passed: {days_p}d {hours_p}h {minutes_p}m {seconds_p}s",
            f'progress: {round(simulation_status["progress"])} %, {simulation_status["state"]}',
            is_first_plot,
            fig if fig_updated else dash.no_update,
        )
    
    elif simulation_status['state'] == 'preparing':
        if trigger == 'uploaded-file-store-simulation' and uploaded_data:
            is_first_plot = False

            X = np.array(uploaded_data["x_deg_mesh"])[0] if meshgrid_type == 'meshgridlatlondeg' else np.array(uploaded_data["x_meter_mesh"])[0]
            Y = np.array(uploaded_data["y_deg_mesh"])[:, 0] if meshgrid_type == 'meshgridlatlondeg' else np.array(uploaded_data["y_meter_mesh"])[:, 0]
            
            height = np.array(uploaded_data["height"])
            igniting_cells = np.zeros_like(height)

            fig = make_subplots(rows=1, cols=2, subplot_titles=["Elevation (edit ignition on this one)", "Simulation"])
            fig.add_trace(go.Heatmap(z=height, x=X, y=Y, colorscale="Geyser", showscale=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(opacity=0)), row=1, col=1)
            fig.add_trace(go.Heatmap(z=igniting_cells, x=X, y=Y, xaxis='x1', yaxis='y1',colorscale="hot"), row=1, col=2)
            fig.update_layout(dragmode="select", xaxis2=dict(matches='x1'), yaxis2=dict(matches='y1'))

            fig_updated = True

        elif trigger == 'simulation-plot' and relayout_data and 'selections' in relayout_data:
            selection = relayout_data['selections'][0]
            if selection.get('xref') == 'x' and selection.get('yref') == 'y':
                x0, x1 = selection['x0'], selection['x1']
                y0, y1 = selection['y0'], selection['y1']
                x_range = sorted([x0, x1])
                y_range = sorted([y0, y1])
                x_indices = np.where((X >= x_range[0]) & (X <= x_range[1]))[0]
                y_indices = np.where((Y >= y_range[0]) & (Y <= y_range[1]))[0]
                igniting_cells[np.ix_(y_indices, x_indices)] = 1

                fig.data[2].z = igniting_cells
                fig.data[2].zmin = 0
                fig.data[2].zmax = 1

                fig_updated = True
        
        else:
            if fig.data and uploaded_data:
                X = np.array(uploaded_data["x_deg_mesh"])[0] if meshgrid_type == 'meshgridlatlondeg' else np.array(uploaded_data["x_meter_mesh"])[0]
                Y = np.array(uploaded_data["y_deg_mesh"])[:, 0] if meshgrid_type == 'meshgridlatlondeg' else np.array(uploaded_data["y_meter_mesh"])[:, 0]
                fig.data[0].x = np.array(uploaded_data["x_deg_mesh"])[0] if meshgrid_type == 'meshgridlatlondeg' else np.array(uploaded_data["x_meter_mesh"])[0]
                fig.data[0].y = np.array(uploaded_data["y_deg_mesh"])[:, 0] if meshgrid_type == 'meshgridlatlondeg' else np.array(uploaded_data["y_meter_mesh"])[:, 0]
                fig.data[2].x = np.array(uploaded_data["x_deg_mesh"])[0] if meshgrid_type == 'meshgridlatlondeg' else np.array(uploaded_data["x_meter_mesh"])[0]
                fig.data[2].y = np.array(uploaded_data["y_deg_mesh"])[:, 0] if meshgrid_type == 'meshgridlatlondeg' else np.array(uploaded_data["y_meter_mesh"])[:, 0]
                fig.data[2].z = igniting_cells

            fig_updated = True

        if fig_updated:
            fig.update_layout(
                    xaxis=dict(scaleanchor="y"),  # Link the x-axis to the y-axis
                    yaxis=dict(scaleanchor="x"),  # Link the y-axis to the x-axis
                    xaxis2=dict(matches='x1'),
                    yaxis2=dict(matches='y1'),
                    dragmode='select',  # Enable selection on the second plot
                )

        return (
            fig if fig_updated else dash.no_update,
            igniting_cells.tolist(),
            dash.no_update,
            dash.no_update,
            is_first_plot,
            fig if fig_updated else dash.no_update,
        )
    
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Update slider range
@app.callback(
    output=[
        Output('frame-slider', 'min'),
        Output('frame-slider', 'max'),
        Output('frame-slider', 'value'),
    ],
    inputs=[Input('sim-frame-store', 'data'),
            Input('simulation-status-store', 'data')],
    prevent_initial_call=True
)
def update_slider_range_and_value(sim_frames, simulation_status):
    if not sim_frames or ("timestamps" not in sim_frames) or (not sim_frames["timestamps"]):
        raise dash.exceptions.PreventUpdate

    total_frames = len(sim_frames["timestamps"])
    if simulation_status['state'] == 'running' or simulation_status['state'] == 'finished':
        return 0, total_frames - 1, total_frames - 1

# simulation buttons logic
@app.callback(
    output=[
        Output('simulate-btn', 'disabled'),
        Output('reset-btn', 'disabled'),
        Output('frame-slider', 'disabled'),
        Output('all-inputs-simulation', 'className')
    ],
    inputs=[
        Input('simulate-btn', 'n_clicks'),
        Input('reset-btn', 'n_clicks'),
        Input('frame-interval', 'n_intervals'),
        Input('simulation-status-store', 'data'),
    ],
    prevent_initial_call=True
)
def handle_simulation_buttons(sim_clicks, reset_clicks, n_intervals, simulation_status):
    trigger = ctx.triggered_id

    if trigger == 'simulate-btn' and sim_clicks > 0:
        return True, True, True, 'all-inputs-simulation-disabled'

    if trigger == 'reset-btn' and reset_clicks > 0:
        return False, True, False, 'all-inputs-simulation'
    
    if simulation_status['state'] == 'running':
        return True, True, True, 'all-inputs-simulation-disabled'
    
    elif simulation_status['state'] == 'finished':
        return True, False, False, 'all-inputs-simulation-disabled'

    raise dash.exceptions.PreventUpdate

# --- Refactored and merged callbacks for simulation control ---

# Run simulation in background (diskcache)
@app.callback(
    Input('simulate-btn', 'n_clicks'),
    [
        State('uploaded-file-store-simulation', 'data'),
        State('store-igniting-cells', 'data'),
        State('settings-store-simulation', 'data'),
        State('simulation-setup-settings-store', 'data'),
        State('simulation-status-store', 'data'),
    ],
    prevent_initial_call=True,
    background=True,
    manager=background_callback_manager,
)
def run_simulation(n_clicks, sim_data, igniting_cells, settings_data, simulation_dt_t_data, simulation_status):
    global simulation_grid, frames_recorder
    if not sim_data or not igniting_cells:
        raise dash.exceptions.PreventUpdate

    if cache_config.acquire_lock("SIMULATION_ONLY_KEY"):
        try:
            simulation_grid.reset()
            frames_recorder.reset()
            simulation_grid.setup(
                np.array(sim_data['height'], dtype=float),
                np.array(sim_data['temperature'], dtype=float),
                np.array(sim_data['fuel_moisture_content'], dtype=float),
                np.array(sim_data['fuel_mass'], dtype=float),
                np.array(sim_data['unburnable_mass'], dtype=float),
                np.array(sim_data['wind_x'], dtype=float),
                np.array(sim_data['wind_y'], dtype=float),
                dict(sim_data['unmod_settings']),
                dict(settings_data),
                dict(sim_data['metadata']),
                np.array(igniting_cells, dtype=float)
            )

            fire_simulation.simulate(
                simulation_grid,
                frames_recorder,
                simulation_dt_t_data['time_step'],
                simulation_dt_t_data['total_time'],
                simulation_dt_t_data['record_interval']
            )
        finally:
            cache_config.release_lock("SIMULATION_ONLY_KEY")
    else:
        print("Simulation already running!!!")

# Collect settings from dynamic inputs or file
@app.callback(
    Output('settings-store-simulation', 'data'),
    [
        Input('uploaded-file-store-simulation', 'data'),
        Input({'type': 'setting-input', 'key': ALL}, 'value')
    ],
    State({'type': 'setting-input', 'key': ALL}, 'id'),
)
def collect_settings(data, values, ids):
    # If file upload triggered, use its mod_settings
    if ctx.triggered_id == 'uploaded-file-store-simulation':
        return data['mod_settings']
    # Otherwise, collect from dynamic inputs
    if not values or not ids:
        raise dash.exceptions.PreventUpdate
    keys = [id_obj['key'] for id_obj in ids]
    return {k: v for k, v in zip(keys, values)}

# Store time/interval settings
@app.callback(
    Output('simulation-setup-settings-store', 'data'),
    [
        Input('simulation-time-step-input', 'value'),
        Input('simulation-total-time-input', 'value'),
        Input('simulation-record-interval-input', 'value')
    ]
)
def set_simulation_its(time_step, total_time, record_interval):
    return {'time_step': time_step, 'total_time': total_time, "record_interval": record_interval}

# Update sim-frame-store with latest frames from cache
@app.callback(
    Output('sim-frame-store', 'data'),
    Input('frame-interval', 'n_intervals'),
    Input('simulate-btn', 'n_clicks'),
    State('sim-frame-store', 'data'),
    prevent_initial_call=True
)
def update_plot_for_simulation(n_intervals, n_clicks, all_data):
    trigger = ctx.triggered_id
    if trigger == 'simulate-btn':
        all_data={
                    "timestamps": [],
                    "fuel_moisture_percentage": {"data": [], 'colormap': 'blues', 'min': float('inf'), 'max': float('-inf')},
                    "temperature_celsius": {"data": [], 'colormap': 'hot', 'min': float('inf'), 'max': float('-inf')},
                    "fire_intensity_kW_m2": {"data": [], 'colormap': 'viridis', 'min': float('inf'), 'max': float('-inf')},
                    "fuel_mass_kg": {"data": [], 'colormap': 'turbid_r', 'min': float('inf'), 'max': float('-inf')},
                    "fire_time_evolution": {"data": [], 'colormap': 'turbo', 'min': float('inf'), 'max': float('-inf')}
                }  # Holds the simulation frames
        
        return all_data
    else:
        if cache.get("progress") < 100:
            newData = cache.get("new_frames")
            print(f'SIMPAGE--                                 fetched timestamps: {newData["timestamps"]}')
            cache.set("new_frames", 
                    {   "timestamps": [],
                        "fuel_moisture_percentage": {"data": [], 'colormap': None, 'min': None, 'max': None},
                        "temperature_celsius": {"data": [], 'colormap': None, 'min': None, 'max': None},
                        "fire_intensity_kW_m2": {"data": [], 'colormap': None, 'min': None, 'max': None},
                        "fuel_mass_kg": {"data": [], 'colormap': None, 'min': None, 'max': None},
                        "fire_time_evolution": {"data": [], 'colormap': None, 'min': None, 'max': None}
                    })

            all_data["timestamps"].extend(newData["timestamps"])

            for key in ["fuel_moisture_percentage", "temperature_celsius", "fire_intensity_kW_m2", "fuel_mass_kg", "fire_time_evolution"]:
                all_data[key]["data"].extend(newData[key]["data"])
                all_data[key]["min"] = newData[key]["min"]
                all_data[key]["max"] = newData[key]["max"]
                all_data[key]["colormap"] = newData[key]["colormap"]
    
            return newData

        else:
            alldata = cache.get("all_frames")
            print(f'SIMPAGE--                          fetched timestamps: {alldata["timestamps"]}')
            return alldata

# Manage simulation status and frame-interval in one callback
@app.callback(
    [
        Output('simulation-status-store', 'data'),
        Output('frame-interval', 'disabled')
    ],
    [
        Input('frame-interval', 'n_intervals'),
        Input('simulate-btn', 'n_clicks'),
        Input('reset-btn', 'n_clicks'),
        Input('upload-wfss-for-simulation', 'contents'),
    ],
    [
        State('simulation-status-store', 'data'),
        State('sim-frame-store', 'data'),
    ],
    prevent_initial_call=True
)
def manage_status_and_interval(n_intervals, n_clicks, reset_n_clicks, upload, simulation_status, latestFrameStore):
    global disableIntervalNextCall
    trigger = ctx.triggered_id
    interval_disabled = dash.no_update

    if disableIntervalNextCall:
        interval_disabled = True
        print(f'SIMPAGE--                                 disabled interval')

    if trigger == 'frame-interval':
        simulation_status['progress'] = float(cache.get("progress") or 0)
        if (simulation_status['progress'] >= 100):
            simulation_status['state'] = 'finished'
            disableIntervalNextCall = True
            print(f'SIMPAGE--                                 state = finished. disabling interval on next call...')
    elif trigger == 'simulate-btn':
        if simulation_status['state'] != 'running':
            simulation_status['state'] = 'running'
            disableIntervalNextCall = False
            interval_disabled = False
    elif trigger == 'upload-wfss-for-simulation':
        simulation_status['state'] = 'preparing'
        interval_disabled = True
    
    elif trigger == 'reset-btn':
        interval_disabled = True
        simulation_status['state'] = 'finished'
        print(f'SIMPAGE--                                 state = finished. interval disabled')

    return simulation_status, interval_disabled
