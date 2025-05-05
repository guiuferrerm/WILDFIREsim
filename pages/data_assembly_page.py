import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app import app
from utils.height_data_array_prep import HGT_to_np_array, prepare_HGT_as_array_data
from utils.npz_file_management import build_npz_file
from utils.dcc_upload_management import read_and_store_dcc_file_at

layout = html.Div([
    html.Div([
        html.Div([
            html.H3("1. Upload File"),
            dcc.Upload(
                id='upload-hgt',
                className='file-upload',
                children=html.Button('Upload .hgt file', className="button"),
                multiple=False,  # Allow only one file for simplicity
                accept='.hgt',  # Only allow CSV and Excel files
            ),
            html.Div(id='output-data-upload', className="error-message"),
            html.Div("No file uploaded", id='upload-status', className="error-message"),
        ], className="box-section"),
    
        # Store the uploaded file data
        dcc.Store(id='uploaded-file-store'),
    
        # Inputs for processing the data
        html.Div([
            html.H3("2. File processing settings"),
            # Row for North inputs
            html.Div([
                html.Label('Origin N (latitude):'),
                dcc.Input(id='originN', type='number', value=37.0, step=0.0001),

                html.Label('Min N (latitude):'),
                dcc.Input(id='minN', type='number', value=37.0, step=0.0001),

                html.Label('Max N (latitude):'),
                dcc.Input(id='maxN', type='number', value=38.0, step=0.0001),
            ], className="input-row"),

            # Row for East inputs
            html.Div([
                html.Label('Origin E (longitude):'),
                dcc.Input(id='originE', type='number', value=-8.0, step=0.0001),

                html.Label('Min E (longitude):'),
                dcc.Input(id='minE', type='number', value=-8.0, step=0.0001),

                html.Label('Max E (longitude):'),
                dcc.Input(id='maxE', type='number', value=-7.0, step=0.0001),
            ], className="input-row"),

            # Row for Arcsec Interval
            html.Div([
                html.Label('Arcsec Interval (30m/arcsec):'),
                dcc.Input(id='arcsecInterval', type='number', value=3, step=0.0001),
            ], className="input-row"),

            # Button to trigger processing
            html.Button('Process Data', id='process-button', className="button"),
            html.Div("Waiting to process", id='state-output', className="error-message")
        ], className="box-section")
    ]),

    # New Data Assembly Section
    html.Div([
        html.H3("3. Data Assembly"),

        # Dropdown for selecting data type (temperature, humidity, etc.)
        html.Div([
            dcc.Input(placeholder="property value",id='data-mod-value', type='number', value=0, step=0.0001),
            html.Label("K", id="data-mod-units"),
            dcc.Dropdown(
                id='data-type-dropdown',
                options=[
                    {'label': 'Temperature', 'value': 'temperature'},
                    {'label': 'Fuel moisture', 'value': 'fuel_moisture_content'},
                    {'label': 'Unburnable Mass', 'value': 'unburnable_mass'},
                    {'label': 'Fuel Mass', 'value': 'fuel_mass'},
                    {'label': 'Wind X', 'value': 'wind_x'},
                    {'label': 'Wind Y', 'value': 'wind_y'},
                ],
                value='temperature',  # Default value
                style={'width': '200px'},
                clearable=False,  # No "reset" button
                searchable=False
            ),
            html.Button('Apply to all', id='mod-all-btn', className="button")
        ], className="input-row"),

        html.Div([
            dcc.Graph(
                id='data-plot',
                config={
                    'displayModeBar': True,
                    'modeBarButtonsToRemove': ['lasso2d'],  # Remove lasso selection tool
                    'displaylogo': False,
                    'scrollZoom': True
                }
            )
        ])
    ], className="box-section"),

    html.Div([
        html.H3("4. Data Download"),
        html.Div([
            dcc.Input(placeholder="file name",id='file-name-input', type='text'),
            dcc.Input(placeholder="setup (project) name",id='setup-name-input', type='text'),
            html.Button('Download', id='download-btn', className="button"),
            dcc.Download(id="download-file")
        ], className='input-row'),

    ], className="box-section"),
])

@app.callback(
    [Output('output-data-upload', 'children'),
     Output('upload-status', 'children'),
     Output('uploaded-file-store', 'data'),
     Output('output-data-upload', 'className'),
     Output('upload-status', 'className')],
    Input('upload-hgt', 'contents'),
    State('upload-hgt', 'filename'),
    prevent_initial_call=True
)
def handle_file_upload(file_contents, filename):
    if file_contents is None:
        return 'No file uploaded yet', '', None, "error-message", "error-message",

    file_data = {
        'file_path': f"/tmp/{filename}",
        'filename': filename
    }

    read_and_store_dcc_file_at(file_contents, filename, f"/tmp/{filename}")

    # Return the appropriate success message and filename
    return f"File uploaded successfully!", f"Uploaded File: {filename}", file_data, "successful-message", "successful-message",

@app.callback(
    [Output('state-output', 'children'),
     Output('state-output', 'className'),
     Output('data-plot', 'figure')],
    [Input('process-button', 'n_clicks'),
     Input('data-type-dropdown', 'value'),
     Input('data-plot', 'relayoutData'),
     Input("mod-all-btn", "n_clicks")],  # Listen for selection events on the graph],
    [State('data-type-dropdown', 'value'),
     State('data-mod-value', 'value'),
     State('uploaded-file-store', 'data'),
     State('originN', 'value'),
     State('minN', 'value'),
     State('maxN', 'value'),
     State('originE', 'value'),
     State('minE', 'value'),
     State('maxE', 'value'),
     State('arcsecInterval', 'value')],
     prevent_initial_call=True
)

def update_plot(n_clicks, dataType, relayout_data, mod_n_clicks,
                            dataTypeState, mod_value, file_data, 
                            originN, minN, maxN,
                            originE, minE, maxE, 
                            arcsecInterval):
    
    global fig, dataArrays, Xmesh, Ymesh, heightData, meterMeshGrid

    trigger_id = ctx.triggered_id

    if trigger_id == 'process-button':
        if n_clicks is None or file_data is None:
            return 'No file uploaded or process button not clicked yet.', "error-message", {}

        try:
            # File processing
            file_path = file_data['file_path']
            height_data = HGT_to_np_array(file_path)
            heightData, meterMeshGrid, arcsecMeshGrid = prepare_HGT_as_array_data(
                height_data, originN, minN, maxN, originE, minE, maxE, arcsecInterval
            )
            
        except Exception as e:
            return f"Error: {str(e)}", "error-message", {}

        # Set up auxiliary data arrays
        dataArrays = {
            "temperature": {"array": np.zeros_like(heightData), "colorscale": "hot", "units": "K"},
            "fuel_moisture_content": {"array": np.zeros_like(heightData), "colorscale": "Blues", "units": "%"},
            "fuel_mass": {"array": np.zeros_like(heightData), "colorscale": "YlGnBu", "units": "Kg/m2"},
            "unburnable_mass": {"array": np.zeros_like(heightData), "colorscale": "Greys", "units": "Kg/m2"},
            "wind_x": {"array": np.zeros_like(heightData), "colorscale": "RdBu", "units": "m/s"},
            "wind_y": {"array": np.zeros_like(heightData), "colorscale": "RdBu", "units": "m/s"},
        }

        Xmesh, Ymesh = arcsecMeshGrid

        # Create figure with initial elevation + placeholder
        fig=0
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Elevation Data", "Placeholder (edit on this one)"])
        
        fig.add_trace(
            go.Heatmap(
                z=heightData,
                x=Xmesh[0],
                y=Ymesh[:, 0],
                colorscale="Geyser",
                colorbar=dict(
                    title="Elevation",
                    y=1.05,          # Closer to the left plot
                    yanchor='top',
                    xref='paper',
                    len=0.55
                )
            ),
            row=1, col=1
        )

        # ADD THIS PLOT TO SHOW THE SELECTION TOOLS --> afterwards fig data 2 bc 1 is 
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],  # avoid affecting scale
                mode='markers',
                marker=dict(opacity=0),
                hoverinfo='skip',
                showlegend=False
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Heatmap(
                z=dataArrays[dataTypeState]["array"],
                x=Xmesh[0],
                y=Ymesh[:, 0],
                colorscale=dataArrays[dataTypeState]["colorscale"],
                xaxis='x1',
                yaxis='y1',
                colorbar=dict(
                    title="Other",
                    y=0.5,          # Closer to the left plot
                    yanchor='top',
                    xref='paper',
                    len=0.55
                )
            ),
            row=1, col=2
        )

        fig.update_layout(
            xaxis=dict(scaleanchor="y"),  # Link the x-axis to the y-axis
            yaxis=dict(scaleanchor="x"),  # Link the y-axis to the x-axis
            xaxis2=dict(matches='x1'),
            yaxis2=dict(matches='y1'),
            dragmode='select'  # Enable selection on the second plot
        )

        return 'Data processed successfully!', "successful-message", fig

    elif trigger_id == 'data-type-dropdown' and fig:
        fig.data[2].colorscale = dataArrays[dataType]["colorscale"]
        fig.data[2].z = dataArrays[dataTypeState]["array"]
        fig.data[2].zmin = np.min(dataArrays[dataTypeState]["array"])
        fig.data[2].zmax = np.max(dataArrays[dataTypeState]["array"])
        return dash.no_update, dash.no_update, fig

    elif trigger_id == 'data-plot' and relayout_data and 'selections' in relayout_data:
        selection = relayout_data['selections'][0]
    
        # Ensure we're responding only to selection on the second subplot
        if selection.get('xref') == 'x2' and selection.get('yref') == 'y2':
            # Get x/y coordinates from selection
            x0, x1 = selection['x0'], selection['x1']
            y0, y1 = selection['y0'], selection['y1']
            
            x_min, x_max = min(x0, x1), max(x0, x1)
            y_min, y_max = min(y0, y1), max(y0, y1)
            
            # Convert coordinate ranges to index masks
            x_coords = Xmesh[0]
            y_coords = Ymesh[:, 0]
            
            x_indices = np.where((x_coords >= x_min) & (x_coords <= x_max))[0]
            y_indices = np.where((y_coords >= y_min) & (y_coords <= y_max))[0]
    
            dataArrays[dataTypeState]["array"][np.ix_(y_indices, x_indices)] = mod_value

            fig.data[2].z = dataArrays[dataTypeState]["array"]
            fig.data[2].zmin = np.min(dataArrays[dataTypeState]["array"])
            fig.data[2].zmax = np.max(dataArrays[dataTypeState]["array"])

            return dash.no_update, dash.no_update, fig
            
        else:
            return dash.no_update, dash.no_update, dash.no_update

    elif trigger_id == 'mod-all-btn':
        dataArrays[dataTypeState]["array"][:, :] = mod_value
        fig.data[2].z = dataArrays[dataTypeState]["array"]
        fig.data[2].zmin = np.min(dataArrays[dataTypeState]["array"])
        fig.data[2].zmax = np.max(dataArrays[dataTypeState]["array"])
        return dash.no_update, dash.no_update, fig
        
    else:
        return dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output('data-mod-units', 'children'),
    Input('data-type-dropdown', 'value'),
    prevent_initial_call=True
)

def update_units_label(dataType):
    global dataArrays
    return dataArrays[dataType]["units"]

@app.callback(
    Output('download-file', 'data'),
    Input('download-btn', 'n_clicks'),
    State('file-name-input', 'value'),
    State('setup-name-input', 'value'),
    prevent_initial_call=True
)

def download_file(n_clicks, fileName, setupName):
    global dataArrays, heightData, Xmesh, Ymesh, meterMeshGrid

    xMeshM, yMeshM = meterMeshGrid
    
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
            "cell_size": round( ((Xmesh[0][-1]-Xmesh[0][0])/len(Xmesh[0]))*3600*30, 3 ),
            "array_dim_x": int(heightData.shape[1]),
            "array_dim_y": int(heightData.shape[0]),

            "water_specific_heat": 4.186,
            "water_latent_heat": 2260.0,
            "water_boiling_temp": 373.15,

            "stefan_boltzmann_ct": 5.670e-11,
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

            "heat_transfer_rate": 0.003,
            "slope_effect_factor": 0.2,
            "wind_effect_constant": 1,
            "fuel_burn_rate": 0.001,
            "heat_loss_factor": 0.999,
            "transfer_heat_loss_factor": 0.7,
            
        },

        "metadata": {
            "version": 1.0,
            "setup_title": setupName,
        },
    }

    file = build_npz_file(arrays)

    return dcc.send_bytes(file, f"{fileName}.wfss")
    