from dash import html, dcc, callback, Input, Output

def get_layout():
    return html.Div([
        html.H1("Дашборд для анализа ДТП и факторов аварийности", style={'textAlign': 'center'}),
        dcc.Tabs(id='tabs', value='tab-statistics', children=[
            dcc.Tab(label='Статистика по регионам', value='tab-statistics', children=[
                html.H2("Статистика ДТП по регионам России"),
                dcc.Dropdown(
                    id='region-dropdown',
                    options=[], 
                    value='Все регионы'
                ),
                html.Div([
                    html.Div([
                        dcc.Graph(id='graph-1-1', style={'width': '48%', 'display': 'inline-block'}),
                        dcc.Graph(id='graph-1-2', style={'width': '48%', 'display': 'inline-block'}),
                        html.Div(
                            children=[
                                dcc.Slider(
                                    id='year-slider', 
                                    min=2015, 
                                    max=2023, 
                                    value=2023, 
                                    marks={str(y): str(y) for y in range(2015, 2024)}, 
                                    tooltip={'placement': 'bottom', 'always_visible': True}, 
                                )
                            ],
                            style={'width':'80%', 'margin':'20px auto'},
                            id='slider-container'
                        )
                    ]),
                    html.Div([
                        dcc.Graph(id='graph-1-3', style={'width': '45%', 'display': 'inline-block','margin': '0 20px'}),
                        dcc.Graph(id='graph-1-4', style={'width': '45%', 'display': 'inline-block','margin': '0 20px'}),
                    ], style={'textAlign': 'center'}),
                ])
            ]),
            dcc.Tab(label='Карта концентрации мест ДТП', value='tab-map', children=[
                html.H2("Места скопления ДТП на карте"),
                dcc.Graph(id='map-accidents', style={'height': '700px'})
            ]),
            dcc.Tab(label='Факторы аварийности', value='tab-factors', children=[
                html.H2("Основные факторы аварийности"),

                html.Div([
                    html.Div([
                        dcc.Graph(id='factor-1', style={'width': '40%', 'display': 'inline-block'}),
                        dcc.Graph(id='factor-2', style={'width': '28%', 'display': 'inline-block'}),
                        dcc.Graph(id='factor-3', style={'width': '28%', 'display': 'inline-block'}),
                    ]),
                    html.Div([
                        dcc.Graph(id='factor-4', style={'width': '40%', 'display': 'inline-block'}),
                        dcc.Graph(id='factor-5', style={'width': '28%', 'display': 'inline-block'}),
                        dcc.Graph(id='factor-6', style={'width': '28%', 'display': 'inline-block'}),
                    ]),
                ])
            ]),
        ])
    ])