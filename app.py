from dash import Dash
from utils.cache_config import background_callback_manager
app = Dash(__name__, suppress_callback_exceptions=True, background_callback_manager=background_callback_manager)
