from dash import callback, Input, Output, dcc, html
import plotly.express as px
import pandas as pd
import numpy as np
from sqlalchemy import func
from db import Session, Accident, Participant, Vehicle, Region
import plotly.graph_objs as go
from sklearn.cluster import DBSCAN

def customize_graph_colors(fig, marker_color='#2171b5'):
    fig.update_traces(marker=dict(color=marker_color))
    return fig

def get_all_regions():
    with Session() as session:
        regions = session.query(Region.region).order_by(Region.region).all()
        return [r[0] for r in regions]

def register_callbacks(app):
    @app.callback(
        Output('region-dropdown', 'options'),
        Input('tabs', 'value')
    )
    def update_region_options(_):
        options = [{'label': 'Все регионы', 'value': 'Все регионы'}]
        for r in get_all_regions():
            options.append({'label': r, 'value': r})
        return options

    @app.callback(
        Output('region-dropdown', 'value'),
        Input('region-dropdown', 'options')
    )
    def set_default_region(options):
        return 'Все регионы'

    @app.callback(
        [
            # Графики вкладки "статистика"
            Output('graph-1-1', 'figure'),
            Output('graph-1-2', 'figure'),
            Output('graph-1-3', 'figure'),
            Output('graph-1-4', 'figure'),
            # Графики вкладки "факторы"
            Output('factor-1', 'figure'),
            Output('factor-2', 'figure'),
            Output('factor-3', 'figure'),
            Output('factor-4', 'figure'),
            Output('factor-5', 'figure'),
            Output('factor-6', 'figure'),
            # Карта
            Output('map-accidents', 'figure'),
        ],
        [Input('tabs', 'value'),
        Input('region-dropdown', 'value'),
        Input('year-slider', 'value')]
    )
    def update_all_graphs(current_tab, selected_region, selected_year):
        # Изначально создаем все графики как пустые, далее- выбор вкладки
        fig1 = fig2 = fig3 = fig4 = go.Figure()
        fig_factor1 = fig_factor2 = fig_factor3 = fig_factor4 = fig_factor5 = fig_factor6 = go.Figure()
        g_map = go.Figure()

        if current_tab == 'tab-statistics':
            # Вкладка "статистика" — создаем 4 графика

            # 1.1. Топ-10 регионов по аварийности
            with Session() as session:
                regions_df = pd.read_sql(session.query(Region.region, Region.population_region).statement, session.bind)
                all_years = list(range(2015, 2024))
                data_by_year = {}
                for year in all_years:
                    accidents_sub = session.query(
                        Accident.region,
                        func.count(Accident.accident_id).label('accidents_count')
                    ).filter(
                        func.extract('year', Accident.datetime) == year
                    ).group_by(Accident.region).all()
                    df_acc = pd.DataFrame(accidents_sub, columns=['region', 'accidents_count'])
                    df_merge = pd.merge(df_acc, regions_df, on='region', how='left')
                    df_merge['accidents_per_100k'] = df_merge['accidents_count'] / df_merge['population_region'] * 100000
                    df_merge['accidents_per_100k'] = df_merge['accidents_per_100k'].round(0)
                    top10 = df_merge.sort_values(by='accidents_per_100k', ascending=False).head(10)[::-1]
                    data_by_year[year] = top10

                if selected_year not in data_by_year:
                    selected_year = 2023
                df = data_by_year[selected_year]
                values = df['accidents_per_100k'].values

                fig1 = px.bar(
                    df,
                    y='region',
                    x='accidents_per_100k',
                    orientation='h',
                    labels={'accidents_per_100k': 'ДТП на 100 000 чел.', 'region': 'Регион'},
                    title='Топ-10 регионов по аварийности ({})'.format(selected_year),
                    color='accidents_per_100k', 
                    color_continuous_scale='Blues')

                fig1.update_layout(
                    xaxis=dict(range=[100, 210], autorange=False)
                )
                fig1.update_traces(
                    marker_line_color='darkblue',
                    marker_line_width=0.2
                )
            

            # 1.2. Топ нарушений ПДД по смертности
            with Session() as session:
                violations_data = session.query(
                    Participant.violations,
                    func.avg(Accident.dead_count).label('avg_dead')
                ).join(Accident, Participant.accident_id == Accident.accident_id
                ).filter(
                    Participant.violations != None,
                    Participant.violations != ''
                ).filter(
            func.extract('year', Accident.datetime) == selected_year
                ).group_by(Participant.violations).order_by(func.avg(Accident.dead_count).desc()).limit(10).all()

                df_violation = pd.DataFrame(violations_data, columns=['violation', 'avg_dead'])
                df_violation = df_violation.sort_values(by='avg_dead', ascending=True)
                df_violation['avg_dead'] = df_violation['avg_dead'] * 100
                df_violation['avg_dead_str'] = df_violation['avg_dead'].round(0)

                norm2 = (df_violation['avg_dead'] - df_violation['avg_dead'].min()) / (df_violation['avg_dead'].max() - df_violation['avg_dead'].min())
                colors2 = [
                '#e5f7ff',
                "#9ecae1",
                "#6baed6",
                "#4292c6",
                "#2171b5",
                "#08519c",
                "#08306b",
            ]
                color_map2 = [colors2[int(val * (len(colors2)-1))] for val in norm2]

                fig2 = px.bar(df_violation, y='violation', x='avg_dead', orientation='h',
                            labels={'violation':'Среднее число смертей', 'avg_dead':'Погибших на 100 ДТП'},
                            title='Топ нарушений ПДД по смертности ({})'.format(selected_year),
                            color_discrete_sequence=color_map2)
                fig2.update_traces(hovertemplate='%{y}<br>В среднем погибло: %{x:.0f} на 100 ДТП')
                fig2.update_traces(marker=dict(color=color_map2))
                fig2.update_traces(
                    marker_line_color='darkblue',
                    marker_line_width=0.2
                )
                fig2.update_layout(
                    yaxis=dict(tick0=0, dtick=1, tickmode='linear')
                )

            # 1.3. Количество ДТП по годам в выбранном регионе
            with Session() as session:
                query = session.query(
                    func.extract('year', Accident.datetime).label('year'),
                    func.count(Accident.accident_id).label('count')
                )
                if selected_region != 'Все регионы':
                    query = query.filter(Accident.region == selected_region)
                query = query.filter(func.extract('year', Accident.datetime) != 2024)
                accidents_years = query.group_by(func.extract('year', Accident.datetime)).order_by(func.extract('year', Accident.datetime)).all()

                df_years = pd.DataFrame(accidents_years, columns=['year', 'count'])
                fig3 = px.bar(df_years, x='year', y='count',
                            labels={'count':'Количество ДТП', 'year':'Год'},
                            title=f'Количество ДТП в регионе: {selected_region}')
                fig3 = customize_graph_colors(fig3)

            # 1.4. Количество погибших в ДТП по годам в выбранном регионе
            with Session() as session:
                query = session.query(
                    func.extract('year', Accident.datetime).label('year'),
                    func.sum(Accident.dead_count).label('dead_sum')
                )
                if selected_region != 'Все регионы':
                    query = query.filter(Accident.region == selected_region)
                query = query.filter(func.extract('year', Accident.datetime) != 2024)
                dead_data = query.group_by(func.extract('year', Accident.datetime)).order_by(func.extract('year', Accident.datetime)).all()

                df_dead_years = pd.DataFrame(dead_data, columns=['year', 'dead_sum'])
                fig4 = px.bar(df_dead_years, x='year', y='dead_sum',
                            labels={'dead_sum':'Погибшие', 'year':'Год'},
                            title=f'Количество погибших в ДТП в регионе: {selected_region}')
                fig4 = customize_graph_colors(fig4)

            return [fig1, fig2, fig3, fig4] + [go.Figure() for _ in range(7)]
        
        # Вкладка "Факторы аварийности"

        elif current_tab == 'tab-factors':
            # 3.1 Зависимость кол-ва пострадавших в ДТП от погоды
            with Session() as session:
                df = pd.read_sql(session.query(
                    Accident.weather,
                    func.avg(Accident.injured_count).label('avg_injured')
                ).filter(Accident.weather != None).group_by(Accident.weather).statement, session.bind)

            df = df.dropna().sort_values(by='avg_injured', ascending=False)
            df['avg_injured_rounded'] = df['avg_injured'].round(0).astype(int)

            num_categories = len(df['weather'])
            height = max(400, num_categories * 40)

            fig_factor1 = px.bar(df, x='weather', y='avg_injured',
                        labels={
                            'avg_injured': 'Пострадавших на 1 ДТП',
                            'weather': 'Погода'
                        },
                        title='Зависимость от погоды')
            fig_factor1.data[0].customdata = df['avg_injured_rounded']
            fig_factor1.update_traces(
                hovertemplate='Количество пострадавших на 1 ДТП: %{customdata}<extra></extra>'
            )
            fig_factor1.update_layout(
                height=height,
                yaxis=dict(
                    tick0=1,
                    dtick=1,
                    range=[1, (df['avg_injured'].max()).round(0)],
                    tickmode='linear'
                ),
            )
            fig_factor1 = customize_graph_colors(fig_factor1)

            # 3.2 Зависимость кол-ва пострадавших в ДТП от влажности дорожного покрытия
            with Session() as session:
                df2 = pd.read_sql(session.query(
                    Accident.wet_road,
                    func.avg(Accident.injured_count).label('avg_injured')
                ).filter(Accident.wet_road != None, Accident.wet_road != 'нет данных').group_by(Accident.wet_road).statement, session.bind)
            df2 = df2.dropna().sort_values(by='avg_injured', ascending=False)
            df2['avg_injured_rounded'] = df2['avg_injured'].round(0).astype(int)

            fig_factor2 = px.bar(
                df2,
                x='wet_road',
                y='avg_injured',
                labels={'avg_injured':'Пострадавших на 1 ДТП', 'wet_road':'Дорожное покрытие'},
                title='Зависимость от влажности дороги'
            )
            fig_factor2.data[0].customdata = df2['avg_injured_rounded']
            fig_factor2.update_traces(
                hovertemplate='Количество пострадавших на 1 ДТП: %{customdata}<extra></extra>'
            )
            fig_factor2.update_layout(
                height=height,
                yaxis=dict(
                    tick0=1,
                    dtick=1,
                    range=[1, 2],
                    tickmode='linear',
                    autorange=False
                )
            )
            fig_factor2 = customize_graph_colors(fig_factor2)


            # 3.3 Зависимость кол-ва смертей на 1 ДТП от наличия освещения
            with Session() as session:
                df3 = pd.read_sql(
                    session.query(
                        Accident.road_light,
                        func.avg(Accident.dead_count).label('dead_count')
                    ).filter(Accident.road_light != None).group_by(Accident.road_light).statement,
                    session.bind
                )
            df3['dead_count'] = df3['dead_count'].round(2)
            df3['road_light'] = df3['road_light'].map({False: 'Нет освещения', True: 'Есть освещение'})
            df3 = df3.dropna().sort_values(by='dead_count', ascending=False)

            fig_factor3 = px.bar(
                df3,
                x='road_light',
                y='dead_count',
                labels={'dead_count':'Смертей на 1 ДТП', 'road_light':'Освещение дороги'},
                title='Зависимость от освещения'
            )
            fig_factor3.update_layout(height=height)
            fig_factor3.update_yaxes(tickvals=[0,1], ticktext=['0','1'], range=[0,1])
            fig_factor3 = customize_graph_colors(fig_factor3)

            # 3.4 Зависимость тяжести ДТП от дефектов дорожного покрытия
            with Session() as session:
                df4 = pd.read_sql(session.query(
                    Accident.road_conditions,
                    Accident.severity
                ).statement, session.bind)
            df4['defects_category'] = df4['road_conditions'].apply(
                lambda x: 'С дефектами' if pd.notnull(x) and 'Дефект' in x else 'Без дефектов'
            )
            df4_grouped = df4.groupby('defects_category').agg({'severity':'mean'}).reset_index()
            df4_grouped = df4_grouped.sort_values(by='severity', ascending=False)

            fig_factor4 = px.bar(df4_grouped, x='defects_category', y='severity', title='Зависимость от дефектов дороги')
            fig_factor4.update_layout(
                xaxis_title='Дорожное покрытие',
                yaxis_title='Средняя тяжесть ДТП',
                annotations=[dict(
                    x=0.95, y=0.95, xref='paper', yref='paper',
                    text='1 — с погибшими<br>0.5 — тяжелые травмы<br>0 — легкие травмы',
                    showarrow=False,
                    align='left'
                )]
            )
            fig_factor4.update_yaxes(tickvals=[0.1, 0.4], ticktext=['Легкие травмы', 'Тяжелые травмы'], range=[0.1,0.4])
            fig_factor4.update_traces(hovertemplate='Средняя тяжесть ДТП: %{y:.2f}<extra></extra>')
            fig_factor4 = customize_graph_colors(fig_factor4)

            # 3.5 Зависимость тяжести ДТП от возраста ТС
            with Session() as session:
                query = session.query(
                    Vehicle.year_auto,
                    func.avg(Accident.severity).label('avg_severity')
                ).select_from(Vehicle).join(Accident, Vehicle.accident_id == Accident.accident_id).filter(
                    Vehicle.year_auto != None,
                    Vehicle.year_auto > 0,
                    Vehicle.year_auto <= 50
                ).group_by(Vehicle.year_auto).order_by(Vehicle.year_auto)

                result = query.all()
                if result:
                    df = pd.DataFrame(result, columns=['year_auto','avg_severity'])
                else:
                    df = pd.DataFrame(columns=['year_auto','avg_severity'])
                fig_factor5 = px.line(df, x='year_auto', y='avg_severity', markers=True,
                                    labels={'year_auto':'Возраст ТС', 'avg_severity':'Средняя тяжесть ДТП'},
                                    title='Зависимость от возраста ТС')
                fig_factor5.update_yaxes(tickvals=[0.23,0.42], ticktext=['Легкие травмы', 'Тяжелые травмы'], range=[0.23,0.42])
                fig_factor5.update_traces(hovertemplate='Средняя тяжесть ДТП: %{y:.2f}<extra></extra>',
                                          line=dict(color='#2171b5'))
                fig_factor5.update_layout(
                    annotations=[dict(
                        x=0.05, y=0.95, xref='paper', yref='paper',
                        text='1 — с погибшими<br>0.5 — тяжелые травмы<br>0 — легкие травмы',
                        showarrow=False,
                        align='left'
                    )]
                )
                """
                # линия тренда
                fig_factor5.add_shape(
                    type='line',
                    x0=df['year_auto'].min(),  
                    y0=0.23,     
                    x1=df['year_auto'].max(),  
                    y1=0.41,      
                    line=dict(color='#e15b62', width=2),
                ) """

            # 3.6 Распределение ДТП по месяцам
            with Session() as session:
                df_months = pd.read_sql(
                    session.query(
                        func.extract('month', Accident.datetime).label('month'),
                        func.count(Accident.accident_id).label('count')
                    ).filter(func.extract('year', Accident.datetime) < 2024)
                    .group_by(func.extract('month', Accident.datetime))
                    .order_by(func.extract('month', Accident.datetime))
                    .statement,
                    session.bind
                )
            df_months['month_num'] = df_months['month'].astype(int)
            df_months['count'] = df_months['count'] / 9
            month_order = ['Дек', 'Янв', 'Фев', 'Март', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя']
            month_numbers_in_order = [12,1,2,3,4,5,6,7,8,9,10,11]
            month_name_map = {number:name for number, name in zip(month_numbers_in_order, month_order)}
            df_months['month_name'] = df_months['month_num'].map(month_name_map)
            df_months['month_name'] = pd.Categorical(df_months['month_name'], categories=month_order, ordered=True)
            df_months = df_months.sort_values('month_name')

            def get_color(m):
                if m in [12,1,2]:
                    return '#d1e5f7'
                elif m in [3,4,5]:
                    return '#86BCE9'
                elif m in [6,7,8]:
                    return '#113B5F' 
                else:
                    return '#2171b5'
            df_months['color'] = df_months['month_num'].apply(get_color)
            df_months['month_name_str'] = df_months['month_name'].astype(str)
            df_months['hover_info'] = df_months['month_name_str'] + '<br>Количество ДТП: ' + df_months['count'].round(0).astype(str)

            fig6 = px.bar(
                df_months,
                x='month_name',
                y='count',
                color='color',
                title='Зависимость от времени года',
                color_discrete_map={
                    '#d1e5f7':'#d1e5f7',
                    '#86BCE9':'#86BCE9',
                    '#113B5F':'#113B5F',
                    '#2171b5':'#2171b5'
                },
                custom_data=['hover_info']
            )
            fig6.update_traces(hovertemplate='%{customdata[0]}<extra></extra>')
            fig6.add_scatter(
                x=df_months['month_name'],
                y=df_months['count'],
                mode='lines+markers',
                line=dict(color='#e15b62'),
                showlegend=True
            )
            fig6.update_layout(
                xaxis_title='Месяц',
                yaxis_title='Количество ДТП в РФ в год',
                showlegend=False
            )
            fig6.update_yaxes(range=[5000,16000])
            fig_factor6 = fig6

            fig6.update_traces(
                    marker_line_color='darkblue',
                    marker_line_width=0.2
                )

            return [go.Figure() for _ in range(4)] + [fig_factor6, fig_factor1, fig_factor3, fig_factor5, fig_factor2, fig_factor4, go.Figure()] 


        # Вкладка "Карта концентрации мест ДТП"

        elif current_tab == 'tab-map':
            # Формула перевода градусов в километры: L = π * R * a / 180
            R = 6378.1  # радиус Земли в км
            L = 0.1 # заданное мною расстояние для образования кластера точек в км (= 100 метров)
            KM_IN_DEGREES = (180*L)/3.14/R   # ='a' в формуле выше
            with Session() as session:
                df_acc = pd.read_sql(session.query(Accident.latitude, Accident.longitude).statement, session.bind)
            if df_acc.empty:
                print("Нет данных для отображения")
            else:
                # Удаляем некорректные координаты (точки скопления ДТП в воде и в лесу) 
                targets = [
                    (60.000191, 30.001399, 0.01),
                    (60.004029, 29.010478, 0.03),  
                    (60.006893, 27.986337, 0.03),
                    (59.005929, 29.993782, 0.1),
                    (58.517481, 31.249124, 0.1)
                    ]
                mask = pd.Series([False] * len(df_acc))

                for t_lat, t_lon, delta in targets:
                    mask |= (
                        (abs(df_acc['latitude'] - t_lat) <= delta) &
                        (abs(df_acc['longitude'] - t_lon) <= delta)
                        )

                df_acc = df_acc[~mask]

                coords = df_acc[['latitude', 'longitude']].to_numpy()

                # Кластеризация с DBSCAN
                db = DBSCAN(eps=KM_IN_DEGREES, min_samples=25).fit(coords)
                labels = db.labels_

                df_acc['cluster'] = labels

                # считаем кол-во ДТП по кластерам
                cluster_counts = df_acc[df_acc['cluster'] != -1].groupby('cluster').size()
                large_clusters = cluster_counts[cluster_counts >= 20].index
                filtered_df = df_acc[df_acc['cluster'].isin(large_clusters)]

                # точки- центры каждого кластера
                cluster_centers = filtered_df.groupby('cluster').agg({
                    'latitude': 'mean',
                    'longitude': 'mean',
                }).reset_index()

                cluster_counts_df = cluster_counts.loc[large_clusters].reset_index()
                cluster_counts_df.columns = ['cluster', 'count']
                clusters_info = pd.merge(cluster_centers, cluster_counts_df, on='cluster')

                fig_map = px.scatter_mapbox(
                    clusters_info,
                    lat='latitude',
                    lon='longitude',
                    size='count',  
                    size_max=100,   
                    color_discrete_sequence=['red'], 
                    hover_name='count', 
                    hover_data={'count': True},
                    zoom=10,
                    mapbox_style='open-street-map'
                )

                fig_map.update_traces(
                    hovertemplate='ДТП: %{hovertext}',
                    hovertext=clusters_info['count']
                )
                fig_map.update_layout(
                    mapbox_center={"lat": 59.9343, "lon": 30.3351},
                    margin={"r":0,"t":0,"l":0,"b":0},
                    uirevision='constant', 
                    dragmode='zoom'
                )

            g_map = fig_map

            return [go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), g_map]

        return [go.Figure() for _ in range(11)]