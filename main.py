import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, State, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from statsmodels.tsa.seasonal import seasonal_decompose
import io
import base64
from datetime import datetime as dt

from prophet import Prophet
from prophet.plot import plot_plotly


def generate_sample_data():
    np.random.seed(42)
    n_customers = 200


    df = pd.read_csv('pharma-data.csv')
    df['Sales'] = df['Quantity'] * df['Price']

    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
    df['Month-Year'] = df['Date'].dt.to_period('M').astype(str)
    df['Avg_Price'] = df['Sales'] / df['Quantity']
    df['Cost'] = df['Price'] * 0.7
    df['Profit'] = df['Sales'] - df['Cost']
    df['Profit_Margin'] = df['Profit'] / df['Sales']

    return df


df = generate_sample_data()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Pharmaceutical Analytics"

card_style = {
    'borderRadius': '10px',
    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
    'padding': '15px',
    'margin': '10px'
}


def calculate_rfm(df):
    current_date = df['Date'].max() + pd.DateOffset(months=1)
    rfm = df.groupby('Customer Name').agg({
        'Date': lambda x: (current_date - x.max()).days,
        'Customer Name': 'count',
        'Sales': 'sum'
    }).rename(columns={
        'Date': 'Recency',
        'Customer Name': 'Frequency',
        'Sales': 'Monetary'
    })


    def safe_qcut(series, q, labels, duplicates='drop'):
        try:
            return pd.qcut(series, q=q, labels=labels, duplicates=duplicates)
        except ValueError:
            return pd.cut(series, bins=len(labels), labels=labels)

    rfm['R_Score'] = safe_qcut(rfm['Recency'], q=3, labels=[3, 2, 1])
    rfm['F_Score'] = safe_qcut(rfm['Frequency'], q=3, labels=[1, 2, 3])
    rfm['M_Score'] = safe_qcut(rfm['Monetary'], q=3, labels=[1, 2, 3])

    rfm['R_Score'] = rfm['R_Score'].astype(int)
    rfm['F_Score'] = rfm['F_Score'].astype(int)
    rfm['M_Score'] = rfm['M_Score'].astype(int)

    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

    # Сегментация клиентов
    segment_bins = [0, 4, 6, 8, 12]  # Измен границы 
    rfm['Segment'] = pd.cut(
        rfm['RFM_Score'],
        bins=segment_bins,
        labels=['Bronze', 'Silver', 'Gold', 'Platinum']
    )

    return rfm


try:
    rfm_data = calculate_rfm(df)
except Exception as e:
    print(f"Ошибка при расчете RFM: {e}")
    # Создаем пустой DataFrame в случае ошибки
    rfm_data = pd.DataFrame(columns=['Recency', 'Frequency', 'Monetary', 'Segment'])


app.layout = dbc.Container([
    dbc.Tabs([
        # Вкладка 1: Анализ продаж
        dbc.Tab(
            label="Анализ продаж",
            children=[
                html.H1("Данные по оптовой и розничной торговле фармацевтических компаний", className="mb-4 text-center"),
                html.P("Дашборд создан на основе датасета https://www.kaggle.com/datasets/krishangupta33/pharmaceutical-company-wholesale-retail-data", className="mb-4 text-center"),

                dbc.Row([
                    dbc.Col([
                        html.Label("Год"),
                        dcc.Dropdown(
                            id='main-year-filter',
                            options=[{'label': str(year), 'value': year} for year in df['Year'].unique()],
                            value=[df['Year'].max()],
                            multi=True
                        )
                    ], width=3),

                    dbc.Col([
                        html.Label("Канал"),
                        dcc.Dropdown(
                            id='main-channel-filter',
                            options=[{'label': channel, 'value': channel} for channel in df['Channel'].unique()],
                            value=df['Channel'].unique().tolist(),
                            multi=True
                        )
                    ], width=3),

                    dbc.Col([
                        html.Label("Класс продукта"),
                        dcc.Dropdown(
                            id='main-product-filter',
                            options=[{'label': pc, 'value': pc} for pc in df['Product Class'].unique()],
                            value=df['Product Class'].unique().tolist(),
                            multi=True
                        )
                    ], width=3)
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H4("Объем продаж", className="card-title"),
                            html.P(id='main-total-sales', className="card-text")
                        ])
                    ], style=card_style), width=3),

                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H4("Средний чек", className="card-title"),
                            html.P(id='main-avg-price', className="card-text")
                        ])
                    ], style=card_style), width=3),

                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H4("Клиенты", className="card-title"),
                            html.P(id='main-unique-customers', className="card-text")
                        ])
                    ], style=card_style), width=3),

                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H4("Топ продукт", className="card-title"),
                            html.P(id='main-top-product', className="card-text")
                        ])
                    ], style=card_style), width=3)
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='main-sales-trend'), width=16),

                ], className="mb-4"),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='main-rfm-analysis'), width=6),
                    dbc.Col(dcc.Graph(id='main-team-performance'), width=6),

                ], className="mb-4"),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='main-sales-forecast'), width=6),
                    dbc.Col(dcc.Graph(id='main-geo-map'), width=6)
                ], className="mb-4"),

            ]
        ),

        # Вкладка 2: Детализация
        dbc.Tab(
            label="Детализация",
            children=[
                html.H2("Детальный анализ", className="mb-4 text-center"),

                dbc.Row([
                    dbc.Col([
                        html.Label("Период"),
                        dcc.DatePickerRange(
                            id='detail-date-range',
                            start_date=df['Date'].min(),
                            end_date=df['Date'].max(),
                            display_format='DD.MM.YYYY'
                        )
                    ], width=3),

                    dbc.Col([
                        html.Label("Группировка 1"),
                        dcc.Dropdown(
                            id='detail-groupby-1',
                            options=[
                                {'label': 'Канал', 'value': 'Channel'},
                                {'label': 'Город', 'value': 'City'},
                                {'label': 'Продукт', 'value': 'Product Name'}
                            ],
                            value='Channel'
                        )
                    ], width=3),

                    dbc.Col([
                        html.Label("Группировка 2"),
                        dcc.Dropdown(
                            id='detail-groupby-2',
                            options=[
                                {'label': 'Класс продукта', 'value': 'Product Class'},
                                {'label': 'Команда', 'value': 'Sales Team'},
                                {'label': 'Менеджер', 'value': 'Manager'}
                            ],
                            value='Product Class'
                        )
                    ], width=3),

                    dbc.Col([
                        html.Label("Метрика"),
                        dcc.Dropdown(
                            id='detail-metric',
                            options=[
                                {'label': 'Продажи', 'value': 'Sales'},
                                {'label': 'Количество', 'value': 'Quantity'},
                                {'label': 'Прибыль', 'value': 'Profit'}
                            ],
                            value='Sales'
                        )
                    ], width=3)
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col([
                        dash_table.DataTable(
                            id='detail-data-table',
                            page_size=10,
                            style_table={'overflowX': 'auto'},
                            style_data_conditional=[]
                        )
                    ], width=12)
                ])
            ]
        )
    ])
], fluid=True)



from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots





@app.callback(
    Output('main-rfm-analysis', 'figure'),
    [Input('main-year-filter', 'value')]
)
def update_rfm_analysis(years):
    if rfm_data.empty:
        return px.pie(title="Нет данных для RFM-анализа")

    try:
        filtered_rfm = rfm_data[rfm_data.index.isin(
            df[df['Year'].isin(years)]['Customer Name'].unique()
        )]
        segment_counts = filtered_rfm['Segment'].value_counts().reset_index()
        return px.pie(segment_counts, values='count', names='Segment',
                      title="RFM Сегментация клиентов")
    except Exception as e:
        print(f"Ошибка при построении RFM: {e}")
        return px.pie(title="Ошибка при построении RFM")


@app.callback(
    Output('main-team-performance', 'figure'),
    [Input('main-year-filter', 'value'),
     Input('main-channel-filter', 'value')]
)
def update_team_performance(years, channels):
    filtered = df[
        (df['Year'].isin(years)) &
        (df['Channel'].isin(channels))
        ]

    team_data = filtered.groupby(['Sales Team', 'Manager'])['Sales'].sum().reset_index()
    return px.bar(team_data, x='Sales Team', y='Sales', color='Manager', title="Продажи по командам")

@app.callback(
    Output('main-sales-forecast', 'figure'),
    [Input('main-year-filter', 'value'),
     Input('main-product-filter', 'value')]
)
def update_forecast(years, products):
    filtered = df[
        (df['Year'].isin(years)) &
        (df['Product Class'].isin(products))
        ]

    ts_data = filtered.groupby('Date')['Sales'].sum().reset_index()
    ts_data.columns = ['ds', 'y']

    model = Prophet(seasonality_mode='multiplicative')
    model.fit(ts_data)
    future = model.make_future_dataframe(periods=6, freq='MS')
    forecast = model.predict(future)

    fig = plot_plotly(model, forecast)
    fig.update_layout(title="6-месячный прогноз продаж")
    return fig

@app.callback(
    [Output('main-total-sales', 'children'),
     Output('main-avg-price', 'children'),
     Output('main-unique-customers', 'children'),
     Output('main-top-product', 'children')],
    [Input('main-year-filter', 'value'),
     Input('main-channel-filter', 'value'),
     Input('main-product-filter', 'value')]
)
def update_main_kpis(years, channels, products):
    filtered = df[
        (df['Year'].isin(years)) &
        (df['Channel'].isin(channels)) &
        (df['Product Class'].isin(products))
        ]

    total_sales = f"${filtered['Sales'].sum():,.0f}"
    avg_price = f"${filtered['Avg_Price'].mean():.2f}"
    unique_customers = filtered['Customer Name'].nunique()
    top_product = filtered.groupby('Product Name')['Sales'].sum().idxmax()

    return [total_sales, avg_price, unique_customers, top_product]


@app.callback(
    Output('main-geo-map', 'figure'),
    [Input('main-year-filter', 'value'),
     Input('main-channel-filter', 'value'),
     Input('main-product-filter', 'value')]
)
def update_geo_map(years, channels, products):
    filtered = df[
        (df['Year'].isin(years)) &
        (df['Channel'].isin(channels)) &
        (df['Product Class'].isin(products))
        ]

    if filtered.empty:
        return px.scatter_geo(title="Нет данных для отображения")

    geo_data = filtered.groupby(['City', 'Latitude', 'Longitude'])['Sales'].sum().reset_index()

    fig = px.scatter_geo(
        geo_data,
        lat='Latitude',
        lon='Longitude',
        size='Sales',
        color='Sales',
        hover_name='City',
        scope='europe',  
        title="География продаж",
        hover_data={'Sales': ':.2f', 'Latitude': False, 'Longitude': False},
        projection='natural earth'
    )

    fig.update_geos(
        showland=True,
        landcolor="lightgray",
        showocean=True,
        oceancolor="azure",
        showcountries=True,
        countrycolor="white"
    )

    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Объем продаж",
            thicknessmode="pixels",
            lenmode="pixels",
            yanchor="top",
            y=1,
            x=0
        )
    )

    return fig


@app.callback(
    Output('main-sales-trend', 'figure'),
    [Input('main-year-filter', 'value'),
     Input('main-channel-filter', 'value'),
     Input('main-product-filter', 'value')]
)
def update_main_trend(years, channels, products):
    filtered = df[
        (df['Year'].isin(years)) &
        (df['Channel'].isin(channels)) &
        (df['Product Class'].isin(products))
        ]

    trend_data = filtered.groupby('Month-Year')['Sales'].sum().reset_index()
    return px.line(trend_data, x='Month-Year', y='Sales', title="Динамика продаж")


@app.callback(
    [Output('detail-data-table', 'data'),
     Output('detail-data-table', 'columns')],
    [Input('detail-date-range', 'start_date'),
     Input('detail-date-range', 'end_date'),
     Input('detail-groupby-1', 'value'),
     Input('detail-groupby-2', 'value'),
     Input('detail-metric', 'value')]
)
def update_detail_table(start_date, end_date, group1, group2, metric):
    filtered = df[
        (df['Date'] >= start_date) &
        (df['Date'] <= end_date)
        ]

    grouped = filtered.groupby([group1, group2])[metric].sum().reset_index()
    columns = [{"name": col, "id": col} for col in grouped.columns]
    return grouped.to_dict('records'), columns


if __name__ == '__main__':
    app.run(debug=True)