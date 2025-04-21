import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app import app
from utils.height_data_array_prep import HGT_to_np_array, prepare_HGT_as_array_data
from utils.npz_file_management import read_and_store_npz_contents
from utils.dcc_upload_management import read_and_store_dcc_file_at


layout = html.Div([
    html.Div([
        html.H3("1. Upload File"),
        dcc.Upload(
            id='upload-wfss',
            className='file-upload',
            children=html.Button('Upload .wfss file', className="button"),
            multiple=False,  # Allow only one file for simplicity
            accept='.wfss',
        ),
        html.Div(id='output-data-upload-revision', className="error-message"),
        html.Div("No file uploaded", id='upload-status-revision', className="error-message"),
        ], className="box-section"),
    
        # Store the uploaded file data
        dcc.Store(id='uploaded-file-store-revision'),

        html.Div([
            html.H3("2. Data Revision"),
    
            # Dropdown for selecting data type (temperature, humidity, etc.)
            html.Div([
                dcc.Input(placeholder="property value",id='data-mod-value-revision', type='number', value=0, step=0.01),
                html.Label("K", id="data-mod-units-revision"),
                dcc.Dropdown(
                    id='data-type-dropdown-revision',
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
                html.Button('Apply to all', id='mod-all-btn-revision', className="button")
            ], className="input-row"),

            html.Div([
                dcc.Graph(id='data-plot-revision')  # Empty plot that will be updated
            ])
        ], className="box-section"),
])

@app.callback(
    [Output('output-data-upload-revision', 'children'),
     Output('upload-status-revision', 'children'),
     Output('uploaded-file-store-revision', 'data'),
     Output('output-data-upload-revision', 'className'),
     Output('upload-status-revision', 'className'),
     Output('data-plot-revision', 'figure'),
     Output('data-mod-units-revision', 'children')],
    Input('upload-wfss', 'contents'),
    Input('data-type-dropdown-revision', 'value'),
    State('upload-wfss', 'filename'),
    State('data-type-dropdown-revision', 'value'),
    prevent_initial_call=True
)

def manage_upload_and_figure(file_contents, visiblePlotTypeInput, filename, visiblePlotType):
    global fig, dataArrays, arrays_dict
    trigger_id = ctx.triggered_id

    if trigger_id == 'upload-wfss':
        if file_contents is None:
            return 'No file uploaded yet', '', None, "error-message", "error-message", dash.no_update, dash.no_update
    
        file_data = {
        'file_path': f"/tmp/{filename}",
        'filename': filename
        }
    
        read_and_store_dcc_file_at(file_contents, filename, f"/tmp/{filename}")

        arrays_dict = read_and_store_npz_contents(file_data["file_path"])
    
        dataArrays = {
            "temperature": {"array": np.copy(arrays_dict["temperature"]), "colorscale": "hot", "units": "K"},
            "fuel_moisture_content": {"array": np.copy(arrays_dict["fuel_moisture_content"]), "colorscale": "Blues", "units": "%"},
            "fuel_mass": {"array": np.copy(arrays_dict["fuel_mass"]), "colorscale": "YlGnBu", "units": "Kg/m2"},
            "unburnable_mass": {"array": np.copy(arrays_dict["unburnable_mass"]), "colorscale": "Greys", "units": "Kg/m2"},
            "wind_x": {"array": np.copy(arrays_dict["wind_x"]), "colorscale": "RdBu", "units": "m/s"},
            "wind_y": {"array": np.copy(arrays_dict["wind_y"]), "colorscale": "RdBu", "units": "m/s"},
        }
    
        Xmesh, Ymesh = arrays_dict["x_deg_mesh"], arrays_dict["y_deg_mesh"]
    
        # Create figure with initial elevation + placeholder
        fig=0
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Elevation Data", "Placeholder"])
        fig.add_trace(
            go.Heatmap(
                z=arrays_dict["height"],
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
    
        fig.add_trace(
            go.Heatmap(
                z=dataArrays[visiblePlotType]["array"],
                x=Xmesh[0],
                y=Ymesh[:, 0],
                colorscale=dataArrays[visiblePlotType]["colorscale"],
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
    
        return f"File uploaded successfully!", f"Uploaded File: {filename}", file_data, "successful-message", "successful-message", fig, dash.no_update

    elif trigger_id == 'data-type-dropdown-revision' and fig:
        fig.data[1].colorscale = dataArrays[visiblePlotTypeInput]["colorscale"]
        fig.data[1].z = dataArrays[visiblePlotTypeInput]["array"]
        fig.data[1].zmin = np.min(dataArrays[visiblePlotTypeInput]["array"])
        fig.data[1].zmax = np.max(dataArrays[visiblePlotTypeInput]["array"])
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, fig, dataArrays[visiblePlotTypeInput]["units"]

    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    