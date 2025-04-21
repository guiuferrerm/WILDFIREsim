import dash
from dash import html, dcc, Output, Input
from app import app
from pages import home_page, simulation_page, data_assembly_page, data_revision_page
from components import navbar

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar.layout,  # Navigation bar
    html.Div(id='page-content')  # Dynamic content
])

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/':
        return home_page.layout
    elif pathname == '/simulation':
        return simulation_page.layout
    elif pathname == '/data_assembly':
        return data_assembly_page.layout
    elif pathname == '/data_revision':
        return data_revision_page.layout
    else:
        return html.H1("404: Page not found", className="text-danger")

if __name__ == '__main__':
    app.run(debug=True)