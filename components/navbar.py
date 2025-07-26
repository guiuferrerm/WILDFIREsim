from dash import html, dcc, Input, Output
from app import app

layout = html.Nav(
    html.Div([
        # Links on the left side
        html.Div([
            dcc.Link('Inici', href='/', id='home-link', className='nav-link'),
            dcc.Link('Simulació', href='/simulation', id='simulation-link', className='nav-link'),
            dcc.Link('Creació de fitxers', href='/data_assembly', id='data-assembly-link', className='nav-link'),
            dcc.Link('Revisió de fitxers', href='/data_revision', id='data-revision-link', className='nav-link'),
        ], className='nav-links'),

        # Title on the right side
        html.H1("WILDFIREsim"),  # Normal text
    ], className='navbar'),
)

# Callback to highlight the active page link
@app.callback(
    [
        Output('home-link', 'className'),
        Output('simulation-link', 'className'),
        Output('data-assembly-link', 'className'),
        Output('data-revision-link', 'className')
    ],
    [Input('url', 'pathname')]
)
def highlight_active_link(pathname):
    # Set the class for each link based on the current pathname
    if pathname == '/':
        return 'nav-link-active', 'nav-link', 'nav-link', 'nav-link'
    elif pathname == '/simulation':
        return 'nav-link', 'nav-link-active', 'nav-link', 'nav-link'
    elif pathname == '/data_assembly':
        return 'nav-link', 'nav-link', 'nav-link-active', 'nav-link'
    elif pathname == '/data_revision':
        return 'nav-link', 'nav-link', 'nav-link', 'nav-link-active'
    else:
        return 'nav-link', 'nav-link', 'nav-link', 'nav-link'
