import dash
from layout import get_layout
from callbacks import register_callbacks
from dash.dependencies import Input, Output


app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Дашборд - аналитика ДТП"
server = app.server

app.layout = get_layout()

register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=False)
