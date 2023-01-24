# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 08:37:36 2022

@author: ZNL
"""
import dash
from dash import dcc, html, callback, Output, Input, State, ctx
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import pandas as pd
import datetime
import pathlib

dash.register_page(__name__,
                   path='/sale-overview',  # '/' is home page and it represents the url
                   name='Sale Overview',  # name of page, commonly used as name of link
                   title='House-Sale-Overview',  # title that appears on browser's tab
                   # image='pg1.png',  # image in the assets folder
                   description='House Sale Overview.'
)

load_figure_template('morph')

# =============================================================================
# Load dataset, data wrangling and preparation
# =============================================================================
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath('data').resolve()
ames = pd.read_csv(DATA_PATH.joinpath('ames_new.csv'))
ames['YrMo_Sold'] = pd.to_datetime(ames['YrMo_Sold'])
ames['Mo_Sold_str'] = ames['Mo_Sold'].astype(str)
# print(ames.shape[0])
# print(ames.YrMo_Sold.dtype)
# print(ames.YrMo_Sold[:5])

### data frames used in graph plotting
# data frames for Sale volume plotting:
vol_df_month = ames.groupby('YrMo_Sold', as_index=False).size().rename(columns={'size': 'volume'})
vol_df_month['year'] = vol_df_month['YrMo_Sold'].dt.year
vol_df_month['month'] = vol_df_month['YrMo_Sold'].dt.month
vol_df_year = ames.groupby('Year_Sold', as_index=False).size().rename(columns={'size': 'volume'})
vol_df_year['Year_str'] = vol_df_year['Year_Sold'].astype(str)

# data frames for gross sale plotting:
gs_df_month = ames.groupby('YrMo_Sold', as_index=False)[['Sale_Price']].sum().rename(columns={'Sale_Price':'Gross_Sale'})
gs_df_month['year'] = gs_df_month['YrMo_Sold'].dt.year
gs_df_month['month'] = gs_df_month['YrMo_Sold'].dt.month
gs_df_year = ames.groupby('Year_Sold', as_index=False)[['Sale_Price']].sum().rename(columns={'Sale_Price':'Gross_Sale'})
gs_df_year['Year_str'] = gs_df_year['Year_Sold'].astype(str)

# data frames for mean sale price plotting:
mp_df_month = ames.groupby('YrMo_Sold', as_index=False)[['Sale_Price']].mean().round(3).rename(columns={'Sale_Price':'Mean_Price'})
mp_df_month['year'] = mp_df_month['YrMo_Sold'].dt.year
mp_df_month['month'] = mp_df_month['YrMo_Sold'].dt.month
mp_df_year = ames.groupby('Year_Sold', as_index=False)[['Sale_Price']].mean().round(3).rename(columns={'Sale_Price':'Mean_Price'})
mp_df_year['Year_str'] = mp_df_year['Year_Sold'].astype(str)

### lists and dictionaries used in graph plotting
year_lst = ['2006', '2007', '2008', '2009', '2010']
all_year_line_color = px.colors.qualitative.G10[0]
year_colors = px.colors.qualitative.Pastel1[:5]
# year_colors = [px.colors.qualitative.Set3[i] for i in [3, 4, 0, 9, 5]]
year_highlight_colors = px.colors.qualitative.Set1[:5]
year_color_dict = dict([(str(i+2006), year_colors[i]) for i in range(5)])
year_hl_color_dict = dict([(str(i+2006), year_highlight_colors[i]) for i in range(5)])

# =============================================================================
# Create all Card/dropdown Components
# =============================================================================
### define functions
def create_month_options(year):
    if year == 'All':
        options = [{'label': 'All', 'value': 'All'}]
    elif year == '2010':
        options = ([{'label': 'All', 'value': 'All'}] 
                   + [{'label': '0'+str(i+1) if i+1 < 10 else str(i+1), 
                       'value': '0'+str(i+1) if i+1 < 10 else str(i+1)} for i in range(7)])
    else: # year != 2010 or 'All'
        options = ([{'label': 'All', 'value': 'All'}] 
                   + [{'label': '0'+str(i+1) if i+1 < 10 else str(i+1), 
                       'value': '0'+str(i+1) if i+1 < 10 else str(i+1)} for i in range(12)])
    return options

def create_indicator(val, ref_val, delta_suffix, fig_width, num_prefix=''):
   fig = (go.Figure(
        go.Indicator(
            mode='number+delta',
            value=val,
            number={'font': {'size': 36, 'color': '#0074D9'},
                    'prefix': num_prefix
                    },
            delta={'reference': ref_val, 'relative': True, 
                    'valueformat':'.2%',
                    # 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'},
                    'font': {'size': 15}, 'position': 'right', 
                    'suffix': delta_suffix}
            )
        )
        .update_layout(height=32, width=fig_width))
   return fig

def create_lineplot_allyears(df, x_var, y_var, 
                             line_color=all_year_line_color,
                             yaxis_title='Monthly Sale Volume'):
    fig = (go.Figure(
        go.Scatter(x=df[x_var], y=df[y_var],
                   mode='lines + markers',
                   # name = 'monthly sale volume',
                   marker=dict(color=line_color,size=6))
        )
        .update_traces(xperiod='M1', xperiodalignment='middle')
        .update_layout(
            xaxis=dict(dtick='M1', tickformat='%b\n%Y', ticklabelmode='period',
                       tickfont=dict(size=10),
                       # title={'text': 'Date'},
                       showgrid=False,
                       showspikes=True,
                       spikethickness=2),
            xaxis_range=[datetime.datetime(2006, 1, 1),
                         datetime.datetime(2010, 8, 31)],
            yaxis=dict(title={'text': yaxis_title}, 
                       showgrid=False, showspikes=True,
                       spikethickness=2),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0)
            ))
    return fig

def create_lineplot_sepyears(df, x_var, y_var, hl_year, 
                             hl_month=None, year_lst=year_lst,
                             year_color_dict=year_color_dict,
                             year_hl_color_dict=year_hl_color_dict,
                             yaxis_title='Monthly Sale Volume'):
    '''
    note: hl_year and hl_month (if assigned) are both in string type
    '''
    fig = go.Figure()
    # plot traces of all un-highlighted years
    for i, y in enumerate(year_lst):
        if y != hl_year:
            # c = year_hl_color_dict[y] if y == year else year_color_dict[y]
            c = year_color_dict[y]
            df_sub = df[df['year'] == int(y)]
            fig.add_scatter(
                x=df_sub[x_var], y=df_sub[y_var],
                mode='lines', name=y, legendrank=i,
                line=dict(color=c, width=3)
                )
    # plot the trace of highlighted year on top        
    df_sub = df[df['year'] == int(hl_year)]
    fig.add_scatter(
        x=df_sub[x_var], y=df_sub[y_var],
        mode='lines', name=hl_year, legendrank=int(hl_year) - 2006,
        line=dict(color=year_hl_color_dict[hl_year], width=4)
        )
    
    fig.update_layout(
        showlegend=True,
        xaxis=dict(showgrid=False,
                   title={'text': 'Month'}, 
                   dtick=1, showspikes=True
                   ),
        yaxis=dict(title={'text': yaxis_title}, 
                   showgrid=False, showspikes=False),
        hovermode='x unified',
        legend=dict(orientation='h',
                    yanchor='top', y=1.03,
                    xanchor='left', x=0.01 
                    ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0)
        )
    # plot the highlighted data point (a month)
    if hl_month is not None:
        df_sub = df[(df['year'] == int(hl_year)) & (df[x_var] == int(hl_month))]
        fig.add_scatter(x=df_sub[x_var], y=df_sub[y_var], mode='markers',
                        name=hl_year + '.' + hl_month,
                        marker=dict(color=year_hl_color_dict[hl_year], size=15,
                                    line={'width': 2, 'color': 'rgb(255,255,204)'}),
                        showlegend=False)
    return fig
# NOT USED:
# line plot: a line for each year, highlight the selected year
# highlight_color = year_hl_color_dict[year]
# line_plot = (go.Figure(line_plot_sep_years)
#              .update_traces(
#                  patch={'line': {'color': highlight_color, 'width': 4}},
#                  selector={'name': year}
#                  ))

def create_bar_plot(df, x_var='Year_str', y_var='volume',
                    color_sequence=['#3283FE'] * 5,
                    yaxis_title='Annual Sale Volume',
                    hover_temp='Year: %{x}<br>Annual Sale Volume: %{y}'): # #3283FE #0099C6
    bar = (px.bar(df, x=x_var, y=y_var,
                  color=x_var,
                  color_discrete_sequence=color_sequence, # ['#0099c6'] * 5,
                  text_auto=True)
           .update_traces(hovertemplate=hover_temp)
           .update_layout(
               showlegend=False,
               yaxis=dict(title={'text': yaxis_title}, 
                          showgrid=False),
               xaxis=dict(title={'text': ''}, showgrid=False),                        
               paper_bgcolor='rgba(0,0,0,0)',
               plot_bgcolor='rgba(0,0,0,0)',
               margin=dict(l=0, r=0, t=0, b=0)
               ))
    return bar

def create_histogram(df, x_var='Sale_Price'):
    fig = (px.histogram(df, x=x_var, marginal='box', opacity=0.7,
                        labels={'Sale_Price': 'Sale Price'},
                        color_discrete_sequence=['#0074D9'] # color of histogram bars
                   )
           .update_layout(
               showlegend=False,
               yaxis=dict(showgrid=False),
               xaxis=dict(showgrid=False),
               annotations=[
                   dict(text='<b>Sale Price Distribution</b> | 2006.01 ~ 2010.07',
                        font={'size':14, 'color': '#0074D9'}, 
                        x=0.95, align='right', xref='paper', y=0.8, yref='paper', 
                        showarrow=False)
                   ],         
               paper_bgcolor='rgba(0,0,0,0)',
               plot_bgcolor='rgba(0,0,0,0)',
               margin=dict(l=0, r=0, t=20, b=0)
               ))
    return fig

def create_box_plot(df, year,
                    hl_month=None, x_var='Mo_Sold_str', y_var='Sale_Price', 
                    yaxis_title='Sale Price',
                    hover_temp='Month: %{x}<br>Sale Price: %{y}',
                    year_color_dict=year_color_dict,
                    year_hl_color_dict=year_hl_color_dict):
    '''
    note: year and hl_month (if assigned) are both in string type
    '''
    n = 7 if year == '2010' else 12
    if not hl_month:
        color_seq = [year_hl_color_dict[year]] * n
    else:
        color_seq = [year_hl_color_dict[year] if (i == int(hl_month)-1) 
                     else year_color_dict[year] for i in range(n)]
    fig = (px.box(df, x=x_var, y=y_var, color=x_var,
                  category_orders={x_var: [str(i+1) for i in range(n)]},
                  points='all', color_discrete_sequence=color_seq)
           .update_traces(
               # jitter=0.2,
               pointpos=0,
               marker=dict(
                   opacity=0.5,
                   line={'width':0.3, 'color': 'rgb(255,255,204)'}),
               hovertemplate=hover_temp
               )
           .update_layout(
               showlegend=False,
               yaxis=dict(title={'text': yaxis_title}, 
                          showgrid=False),
               xaxis=dict(title={'text': 'Month'}, dtick=1, showgrid=False),
               annotations=[
                   dict(text='<b>{} Monthly Sale Price Distribution</b>'.format(year),
                        font={'size':14, 'color': year_hl_color_dict[year]}, 
    #                     bgcolor='rgba(233,233,233,0.3)',
    #                     bordercolor='#ff7f0e', borderwidth=0.5, borderpad=3,
                        x=0.05, align='left', xref='paper', y=0.9, yref='paper', 
                        showarrow=False)
               ],
               paper_bgcolor='rgba(0,0,0,0)',
               plot_bgcolor='rgba(0,0,0,0)',
               margin=dict(l=0, r=0, t=0, b=0)
               ))
    return fig

def create_ridgeline_plot(df, hl_year=None, x_var='Sale_Price', year_lst=year_lst,
                          year_color_dict=year_color_dict,
                          year_hl_color_dict=year_hl_color_dict):
    fig = go.Figure()
    for y in year_lst:
        sub_df = df[df['Year_Sold'] == int(y)]
        if not hl_year:
            color = '#3283FE'
        else:
            color = year_hl_color_dict[y] if y == hl_year else year_color_dict[y]
        fig.add_trace(
            go.Violin(x=sub_df[x_var], line_color=color, name=y)
            )
    fig.update_traces(
        orientation='h', side='positive', width=2, 
        points=False, meanline_visible=True
        )
    
    fig.update_layout(
        showlegend=False,
        yaxis=dict(showgrid=False),
        xaxis=dict(zeroline=False, showgrid=False),
        annotations=[
            dict(text='<b>Sale Price Distribution for each year</b>',
                 font={'size':14, 'color': '#3283FE'}, 
#                     bgcolor='rgba(233,233,233,0.3)',
#                     bordercolor='#ff7f0e', borderwidth=0.5, borderpad=3,
                 x=0.95, align='right', xref='paper', y=1.02, yref='paper', 
                 showarrow=False)
            ],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0, pad=0)
        )
    return fig

### layout components
card_sale_volume = dbc.Card([
    dbc.CardHeader([
        # select year month
        dbc.Form(
            dbc.Row([
                dbc.Label('Select Year', width='auto', 
                          class_name='text-info fw-bold my-0 py-0'),
                dbc.Col([
                    dbc.Select(id='dp-select-year',
                               options=[
                                   {'label': val, 'value': val}
                                   for val in ['All', '2006', '2007', '2008', '2009', '2010']
                                   ],
                               value='All',
                               class_name='p-1 my-0 text-info',
                               style={'font-size': '15px'}
                               )],
                    width=2,
                    class_name='me-1 my-0 py-0',
                ),
                dbc.Label('Select Month', width='auto',
                          class_name='text-info fw-bold my-0 py-0'),
                dbc.Col([
                    dbc.Select(id='dp-select-month',
                               options=[],
                               value='All',
                               required=True,
                               class_name='p-1 my-0 text-info',
                               style={'font-size': '15px'}
                               )],
                    width=2,
                    class_name='me-2 my-0 py-0',
                ),
                dbc.Col(
                    html.H4(html.I(className='bi bi-arrow-right-circle me-2 text-info')), # bi bi-caret-right-fill
                    width='auto',
                    class_name='my-0 pt-2 pb-0 mb-0'
                    ),
                dbc.Col([
                    dbc.Button(['Visualize'], 
                               id='button-visualize-volume',
                               active=True,
                               class_name='fw-bold text-white py-2 px-3 mx-1 my-0 bg-warning',
                               n_clicks=0,
                                style={
                                    'font-size': '15px',
                                    # 'backgroundColor':'#FF9900'
                                    }
                               )],
                    class_name='my-0 py-0',
                    width='auto'
                    )
                ],
                class_name='my-0 py-0 align-items-center',
            ),
            class_name='my-0 py-0'
        )
        ],
        class_name='py-0'),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                # icon and title Sale Volume
                html.H3([html.I(className='bi bi-house-fill me-3'), 'Sale Volume'], 
                        className='text-nowrap ms-2 text-info fw-bold'
                        )
                ],
                class_name='my-0 ms-5 pt-2',
                width=3),
            dbc.Col([
                dbc.Label([], 
                    id='year-month-label',
                    class_name='fw-bold pt-2 pb-0 mb-0')
                ],
                class_name='my-0 p-0',
                width=2),
            dbc.Col([
                # go.Indicator
                dcc.Graph(id='volume-indicator', figure={},
                          className='p-0 h-100 fw-bold')
                ],
                class_name='my-0 p-0',
                width=3)
            ],
            class_name='align-items-center',),
        dbc.Row([
            # graphs
            dbc.Col([
                # line graph
                dcc.Graph(id='line-sale-volume', figure={},
                          className='p-0 h-100',
                          )
                
                ],
                width=8),
            dbc.Col([
                # bar graph
                dcc.Graph(id='bar-sale-volume-year', figure={},
                          className='p-0 h-100',
                          )
                ],
                width=4)
            ],
            style={'height':'18rem'},
            # class_name='h-100'
            )
        ],
        class_name='py-2')
    ],
    class_name='mt-2',
    style={'height':'25rem'})

card_sale_gross = dbc.Card([
    dbc.CardHeader([
        # select year month
        dbc.Form(
            dbc.Row([
                dbc.Label('Select Year', width='auto', 
                          class_name='text-info fw-bold my-0 py-0'),
                dbc.Col([
                    dbc.Select(id='dp-select-year-2',
                               options=[
                                   {'label': val, 'value': val}
                                   for val in ['All', '2006', '2007', '2008', '2009', '2010']
                                   ],
                               value='All',
                               class_name='p-1 my-0 text-info',
                               style={'font-size': '15px'}
                               )],
                    width=2,
                    class_name='me-1 my-0 py-0',
                ),
                dbc.Label('Select Month', width='auto',
                          class_name='text-info fw-bold my-0 py-0'),
                dbc.Col([
                    dbc.Select(id='dp-select-month-2',
                               options=[],
                               value='All',
                               required=True,
                               class_name='p-1 my-0 text-info',
                               style={'font-size': '15px'}
                               )],
                    width=2,
                    class_name='me-2 my-0 py-0',
                ),
                dbc.Col(
                    html.H4(html.I(className='bi bi-arrow-right-circle me-2 text-info')), # bi bi-caret-right-fill
                    width='auto',
                    class_name='my-0 pt-2 pb-0 mb-0'
                    ),
                dbc.Col([
                    dbc.Button(['Visualize'], 
                               id='button-visualize-gross-sale',
                               active=True,
                               class_name='fw-bold text-white py-2 px-3 mx-1 my-0 bg-warning',
                               n_clicks=0,
                                style={
                                    'font-size': '15px',
                                    # 'backgroundColor':'#FF9900'
                                    }
                               )],
                    class_name='my-0 py-0',
                    width='auto'
                    )
                ],
                class_name='my-0 py-0 align-items-center',
            ),
            class_name='my-0 py-0'
        )
        ],
        class_name='py-0'),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                # icon and title Sale Volume
                html.H3([html.I(className='bi bi-cash-coin me-3'), 'Gross Sale'], 
                        className='text-nowrap ms-2 text-info fw-bold'
                        )
                ],
                class_name='my-0 ms-5 pt-2',
                width=3),
            dbc.Col([
                dbc.Label([], 
                    id='year-month-label-2',
                    class_name='fw-bold pt-2 pb-0 mb-0')
                ],
                class_name='my-0 p-0',
                width=2),
            dbc.Col([
                # go.Indicator
                dcc.Graph(id='gross-sale-indicator', 
                          figure={},
                          className='p-0 h-100 fw-bold')
                ],
                class_name='my-0 p-0',
                width=3)
            ],
            class_name='align-items-center',),
        dbc.Row([
            # graphs
            dbc.Col([
                # line graph
                dcc.Graph(id='line-gross-sale', figure={},
                          className='p-0 h-100',
                          )
                
                ],
                width=8),
            dbc.Col([
                # bar graph
                dcc.Graph(id='bar-gross-sale-year', figure={},
                          className='p-0 h-100',
                          )
                ],
                width=4)
            ],
            style={'height':'18rem'},
            # class_name='h-100'
            )
        ],
        class_name='py-2')
    ],
    class_name='mt-3',
    style={'height':'25rem'})

card_sale_price = dbc.Card([
    dbc.CardHeader([
        # select year month
        dbc.Form(
            dbc.Row([
                dbc.Label('Select Year', width='auto', 
                          class_name='text-info fw-bold my-0 py-0'),
                dbc.Col([
                    dbc.Select(id='dp-select-year-3',
                               options=[
                                   {'label': val, 'value': val}
                                   for val in ['All', '2006', '2007', '2008', '2009', '2010']
                                   ],
                               value='All',
                               class_name='p-1 my-0 text-info',
                               style={'font-size': '15px'}
                               )],
                    width=2,
                    class_name='me-1 my-0 py-0',
                ),
                dbc.Label('Select Month', width='auto',
                          class_name='text-info fw-bold my-0 py-0'),
                dbc.Col([
                    dbc.Select(id='dp-select-month-3',
                               options=[],
                               value='All',
                               required=True,
                               class_name='p-1 my-0 text-info',
                               style={'font-size': '15px'}
                               )],
                    width=2,
                    class_name='me-2 my-0 py-0',
                ),
                dbc.Col(
                    html.H4(html.I(className='bi bi-arrow-right-circle me-2 text-info')), # bi bi-caret-right-fill
                    width='auto',
                    class_name='my-0 pt-2 pb-0 mb-0'
                    ),
                dbc.Col([
                    dbc.Button(['Visualize'], 
                               id='button-visualize-sale-price',
                               active=True,
                               class_name='fw-bold text-white py-2 px-3 mx-1 my-0 bg-warning',
                               n_clicks=0,
                               style={
                                   'font-size': '15px',
                                    # 'backgroundColor':'#FF9900'
                                    }
                               )],
                    class_name='my-0 py-0',
                    width='auto'
                    )
                ],
                class_name='my-0 py-0 align-items-center',
            ),
            class_name='my-0 py-0'
        )
        ],
        class_name='py-0'),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                # icon and title Sale Volume
                html.H3([html.I(className='bi bi-tag-fill me-3'), 'Mean House Price'], 
                        className='text-nowrap ms-2 text-info fw-bold'
                        )
                ],
                class_name='my-0 ms-5 pt-2',
                width=3),
            dbc.Col([
                dbc.Label([], 
                    id='year-month-label-3',
                    class_name='fw-bold pt-2 pb-0 mb-0')
                ],
                class_name='ms-5 my-0 p-0',
                width=2),
            dbc.Col([
                # go.Indicator
                dcc.Graph(id='mean-price-indicator', 
                          figure=create_indicator(val=100000, ref_val=90000, delta_suffix=' vs. last year', fig_width=500, num_prefix='$'),
                          className='p-0 h-100 fw-bold')
                ],
                class_name='my-0 p-0',
                width=3)
            ],
            class_name='align-items-center',),
        dbc.Row([
            # graphs
            dbc.Col([
                # line graph
                dcc.Graph(id='line-mean-price', figure={},
                          className='p-0 h-100',
                          )
                
                ],
                width=8),
            dbc.Col([
                # bar graph
                dcc.Graph(id='bar-mean-price-year', figure={},
                          className='p-0 h-100',
                          )
                ],
                width=4)
            ],
            style={'height':'18rem'},
            # class_name='h-100'
            ),
        dbc.Row([
            dbc.Col([
                # histogram or boxplot
                dcc.Graph(id='hist-box-sale-price', figure={},
                          className='p-0 h-100',
                          )
                ],
                width=8),
            dbc.Col([
                # histogram
                dcc.Graph(id='hist-sale-price-allyears', figure={},
                          className='p-0 h-100',
                          )
                ],
                width=4)
            ],
            style={'height':'20rem'})
        ],
        class_name='py-2')
    ],
    class_name='mt-3',
    style={'height':'50rem'})

# =============================================================================
# Layout
# =============================================================================
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3('House Sale Overview', className='fw-bold my-0', 
                    style={'display':'inline-block'}),
            html.P('Ames, Iowa, 2006-2010', className='ms-3 my-0', 
                   style={'display':'inline-block'})
            ],
            class_name='fw-bold text-center mt-3 mb-1 border-2 border-bottom shadow-sm',
            width={'size': 7, 'offset': 1}),
        dbc.Col([
            dbc.Button(['About this Page'],
                       id='button-about-pg-2',
                       n_clicks=0, 
                       active=True, 
                       class_name='bg-success text-white fw-bold fs-6 py-1 my-0')
            ],
            class_name='mt-3 mb-1',
            width=2),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle('About This Page')),
            dbc.ModalBody([
                dcc.Markdown([
                    '''This page acts as an **interactive sale report** to look into three important
                      sale-related parameters over time: **sale volumn, gross sale, and sale price (both mean price and price distribution)**.
                      \n- Only variable 'Sale_Price' and 'YrMo_Sold' (year-month for each sale) in the dataset are used for sale overview in this 
                      page. \n- Use Year and Month selecting functions for each parameter to flexibly check specific value for the year or month 
                      you are interested in.\n- Use 'Visualize' button right next to the Year-Month options to activate data visualization
                      and show the results.\n- Try to get insights from the data. Is there any trend or pattern in those parameters for 
                      all the recorded sale data?'''
                    ]),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Img(src='assets/Picture11.png',
                                     height='120px',
                                     className='')
                            ],
                            width=3),
                        dbc.Col([
                            html.Img(src='assets/Picture12.png',
                                      height='120px',
                                      className='')
                            ],
                            width=3)
                        ],
                        class_name='justify-content-around')
                    ],
                    className=''),
                ]),
            dbc.ModalFooter(
                html.P('Part 1: House Sale Overview'))
            ],
            id='modal-about-pg-2',
            size='lg',
            scrollable=True,
            is_open=False)
        ], 
        justify='center'),
    dbc.Row([
        dbc.Col([
            card_sale_volume
            ], 
            width=11)
        ], 
        justify='center'),
    dbc.Row([
        dbc.Col([
            card_sale_gross
            ], 
            width=11)
        ], 
        justify='center'),
    dbc.Row([
        dbc.Col([
            card_sale_price
            ], 
            width=11)
        ], 
        justify='center')
    ],
    fluid=True
    )

# =============================================================================
# Callbacks
# =============================================================================
# about this page modal
@callback(
    Output('modal-about-pg-2', 'is_open'),
    [Input('button-about-pg-2', 'n_clicks')],
    [State('modal-about-pg-2', 'is_open')]
)
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

# month dropdown options controled by year dropdown value
@callback(
    Output('dp-select-month', 'options'),
    # Output('dp-select-month', 'disabled'),
    Input('dp-select-year', 'value'),
    )
def toggle_month_options(year):
    options = create_month_options(year=year)
    return options

@callback(
    Output('dp-select-month-2', 'options'),
    Input('dp-select-year-2', 'value'),
    )
def toggle_month_options_2(year):
    options = create_month_options(year=year)
    return options

@callback(
    Output('dp-select-month-3', 'options'),
    Input('dp-select-year-3', 'value'),
    )
def toggle_month_options_3(year):
    options = create_month_options(year=year)
    return options

# update all label and graph component in sale volume card
@callback(
    Output('year-month-label', 'children'),
    Output('volume-indicator', 'figure'),
    Output('line-sale-volume', 'figure'),
    Output('bar-sale-volume-year', 'figure'),
    Input('dp-select-year', 'value'),
    Input('dp-select-month', 'value'),
    Input('button-visualize-volume', 'n_clicks')
    )
def update_card_volume(year, month, n):
    '''
    note: argument year and month are both in string type
    '''
    ### define values and graphs
    # date label
    date_label_allyears = '2006.01 ~ 2010.07'
    # volume indicator
    vol_indic_allyears = create_indicator(val=ames.shape[0], 
                                          ref_val=ames.shape[0], 
                                          delta_suffix=' vs. ------', 
                                          fig_width=300, 
                                          num_prefix='')
    # line plot: all years one line
    line_plot_allyears = create_lineplot_allyears(df=vol_df_month, 
                                                  x_var='YrMo_Sold', 
                                                  y_var='volume', 
                                                  line_color=all_year_line_color)
    # bar plot: call function create_bar_plot()
    bar_plot_allyears = create_bar_plot(df=vol_df_year, color_sequence=['#3283FE'] * 5)
    
    ### initialize all label and graphs in the card
    if n == 0:
        return date_label_allyears, vol_indic_allyears, line_plot_allyears, bar_plot_allyears
    ### update all label and graphs in the card with button
    elif 'button-visualize-volume' == ctx.triggered_id:
        date_label, vol_indic, line_plot, bar_plot = '', {}, {}, {}
        # graphs for all years
        if year == 'All':
            date_label = date_label_allyears
            vol_indic = vol_indic_allyears
            line_plot = line_plot_allyears
            bar_plot = bar_plot_allyears
        
        # graphs for a specific year
        elif month == 'All':
            # year-month-label:
            date_label = ('{}.01 ~ {}.12'.format(year, year) if year != '2010' 
                          else '2010.01 ~ 2010.07')
            # volume indicator
            indic_val = vol_df_year[vol_df_year['Year_Sold'] == int(year)]['volume'].values[0]
            ref_val = 0
            if year == '2006':
                ref_val = indic_val
            elif year == '2010': # use total volume of first 7 months in 2009
                ref_val = vol_df_month[(vol_df_month['year'] == 2009) & (vol_df_month['month'].lt(8))]['volume'].sum()
            else:
                ref_val = vol_df_year[vol_df_year['Year_Sold'] == int(year) - 1].volume.values[0]
            
            suffix_text = ' vs. first 7 months of 2009' if year == '2010' else ' vs. previous year'
            
            vol_indic = create_indicator(val=indic_val, 
                                         ref_val=ref_val, 
                                         delta_suffix=suffix_text, 
                                         fig_width=500, 
                                         num_prefix='')
            # line plot: a line for each year
            line_plot = create_lineplot_sepyears(df=vol_df_month, 
                                                 x_var='month', y_var='volume', 
                                                 hl_year=year,
                                                 hl_month=None, year_lst=year_lst,
                                                 year_color_dict=year_color_dict,
                                                 year_hl_color_dict=year_hl_color_dict,
                                                 yaxis_title='Monthly Sale Volume')
            
            # bar graph: all years, highlight the selected year
            color_seq = [year_hl_color_dict[y] if y == year 
                         else year_color_dict[y] for y in year_lst]
            bar_plot = create_bar_plot(df=vol_df_year, color_sequence=color_seq)
            
        # graph for a specific month
        else:
            date_label = year + '.' + month
            # volume indicator
            indic_val = vol_df_month[(vol_df_month['year'] == int(year)) 
                                     & (vol_df_month['month'] == int(month))]['volume'].values[0]
            ref_val = 0
            if month == '01':
                ref_val = indic_val
            else:
                ref_val = vol_df_month[(vol_df_month['year'] == int(year)) 
                                       & (vol_df_month['month'] == int(month)-1)]['volume'].values[0]
            
            suffix_text = ' vs. previous month'
            vol_indic = create_indicator(val=indic_val, 
                                         ref_val=ref_val, 
                                         delta_suffix=suffix_text, 
                                         fig_width=400, 
                                         num_prefix='')
            # line plot: a line for each year, highlight a point
            line_plot = create_lineplot_sepyears(df=vol_df_month, 
                                                 x_var='month', y_var='volume', 
                                                 hl_year=year,
                                                 hl_month=month, year_lst=year_lst,
                                                 year_color_dict=year_color_dict,
                                                 year_hl_color_dict=year_hl_color_dict,
                                                 yaxis_title='Monthly Sale Volume')
            # bar graph: all years, highlight the selected year
            color_seq = [year_hl_color_dict[y] if y == year 
                         else year_color_dict[y] for y in year_lst]
            bar_plot = create_bar_plot(df=vol_df_year, color_sequence=color_seq)
        
        return date_label, vol_indic, line_plot, bar_plot
    else:
        return dash.no_update
 
# update all label and graph component in gross sale card
@callback(
    Output('year-month-label-2', 'children'),
    Output('gross-sale-indicator', 'figure'),
    Output('line-gross-sale', 'figure'),
    Output('bar-gross-sale-year', 'figure'),
    Input('dp-select-year-2', 'value'),
    Input('dp-select-month-2', 'value'),
    Input('button-visualize-gross-sale', 'n_clicks')
    )
def update_card_gross_sale(year, month, n):
    '''
    note: argument year and month are both in string type
    '''
    ### define values and graphs
    # date label
    date_label_allyears = '2006.01 ~ 2010.07'
    # volume indicator
    gs_indic_allyears = create_indicator(val=ames['Sale_Price'].sum(), 
                                          ref_val=ames['Sale_Price'].sum(), 
                                          delta_suffix=' vs. ------', 
                                          fig_width=300, 
                                          num_prefix='$')
    # line plot: all years one line
    # gs_df_month, gs_df_year, Gross_Sale
    line_plot_allyears = create_lineplot_allyears(df=gs_df_month, 
                                                  x_var='YrMo_Sold', 
                                                  y_var='Gross_Sale', 
                                                  line_color=all_year_line_color,
                                                  yaxis_title='Monthly Gross Sale')
    # bar plot: call function create_bar_plot()
    bar_plot_allyears = create_bar_plot(df=gs_df_year, y_var='Gross_Sale',
                                        yaxis_title='Annual Gross Sale',
                                        hover_temp='Year: %{x}<br>Annual Gross Sale: %{y}')
    ### initialize all label and graphs in the card
    if n == 0:
        return date_label_allyears, gs_indic_allyears, line_plot_allyears, bar_plot_allyears
    ### update all label and graphs in the card with button
    elif 'button-visualize-gross-sale' == ctx.triggered_id:
        date_label, gs_indic, line_plot, bar_plot = '', {}, {}, {}
        # graphs for all years
        if year == 'All':
            date_label = date_label_allyears
            gs_indic = gs_indic_allyears
            line_plot = line_plot_allyears
            bar_plot = bar_plot_allyears
        
        # graphs for a specific year
        elif month == 'All':
            # year-month-label:
            date_label = ('{}.01 ~ {}.12'.format(year, year) if year != '2010' 
                          else '2010.01 ~ 2010.07')
            # gross sale indicator
            indic_val = gs_df_year[gs_df_year['Year_Sold'] == int(year)]['Gross_Sale'].values[0]
            ref_val = 0
            if year == '2006':
                ref_val = indic_val
            elif year == '2010': # use total volume of first 7 months in 2009
                ref_val = gs_df_month[(gs_df_month['year'] == 2009) & (gs_df_month['month'].lt(8))]['Gross_Sale'].sum()
            else:
                ref_val = gs_df_year[gs_df_year['Year_Sold'] == int(year) - 1].Gross_Sale.values[0]
            
            suffix_text = ' vs. first 7 months of 2009' if year == '2010' else ' vs. previous year'
            
            gs_indic = create_indicator(val=indic_val, 
                                        ref_val=ref_val, 
                                        delta_suffix=suffix_text, 
                                        fig_width=500, 
                                        num_prefix='$')
            # line plot: a line for each year
            line_plot = create_lineplot_sepyears(df=gs_df_month, 
                                                 x_var='month', y_var='Gross_Sale', 
                                                 hl_year=year,
                                                 hl_month=None, year_lst=year_lst,
                                                 year_color_dict=year_color_dict,
                                                 year_hl_color_dict=year_hl_color_dict,
                                                 yaxis_title='Monthly Gross Sale')
            
            # bar graph: all years, highlight the selected year
            color_seq = [year_hl_color_dict[y] if y == year 
                         else year_color_dict[y] for y in year_lst]
            bar_plot = create_bar_plot(df=gs_df_year, y_var='Gross_Sale', 
                                       color_sequence=color_seq,
                                       yaxis_title='Annual Gross Sale',
                                       hover_temp='Year: %{x}<br>Annual Gross Sale: %{y}')   
        # graph for a specific month
        else:
            date_label = year + '.' + month
            # volume indicator
            indic_val = gs_df_month[(gs_df_month['year'] == int(year))
                                    & (gs_df_month['month'] == int(month))]['Gross_Sale'].values[0]
            ref_val = 0
            if month == '01':
                ref_val = indic_val
            else:
                ref_val = gs_df_month[(gs_df_month['year'] == int(year))
                                      & (gs_df_month['month'] == int(month)-1)]['Gross_Sale'].values[0]
            
            suffix_text = ' vs. previous month'
            gs_indic = create_indicator(val=indic_val, 
                                        ref_val=ref_val, 
                                        delta_suffix=suffix_text, 
                                        fig_width=400, 
                                        num_prefix='$')
            # line plot: a line for each year, highlight a point
            line_plot = create_lineplot_sepyears(df=gs_df_month, 
                                                 x_var='month', y_var='Gross_Sale', 
                                                 hl_year=year,
                                                 hl_month=month, year_lst=year_lst,
                                                 year_color_dict=year_color_dict,
                                                 year_hl_color_dict=year_hl_color_dict,
                                                 yaxis_title='Monthly Gross Sale')
            # bar graph: all years, highlight the selected year
            color_seq = [year_hl_color_dict[y] if y == year 
                         else year_color_dict[y] for y in year_lst]
            bar_plot = create_bar_plot(df=gs_df_year, y_var='Gross_Sale',
                                       color_sequence=color_seq,
                                       yaxis_title='Annual Gross Sale',
                                       hover_temp='Year: %{x}<br>Annual Gross Sale: %{y}')
        
        return date_label, gs_indic, line_plot, bar_plot
    else:
        return dash.no_update

# update all label and graph component in sale price card
@callback(
    Output('year-month-label-3', 'children'),
    Output('mean-price-indicator', 'figure'),
    Output('line-mean-price', 'figure'),
    Output('bar-mean-price-year', 'figure'),
    Output('hist-box-sale-price', 'figure'),
    Output('hist-sale-price-allyears', 'figure'),
    Input('dp-select-year-3', 'value'),
    Input('dp-select-month-3', 'value'),
    Input('button-visualize-sale-price', 'n_clicks')
    )
def update_card_sale_Price(year, month, n):
    '''
    note: argument year and month are both in string type
    '''
    ### define values and graphs
    # date label
    date_label_allyears = '2006.01 ~ 2010.07'
    # volume indicator
    mp_indic_allyears = create_indicator(val=ames['Sale_Price'].mean().round(3), 
                                         ref_val=ames['Sale_Price'].mean().round(3), 
                                         delta_suffix=' vs. ------', 
                                         fig_width=300, 
                                         num_prefix='$')
    # line plot: all years one line
    # mp_df_month, mp_df_year, Mean_Price
    line_plot_allyears = create_lineplot_allyears(df=mp_df_month, 
                                                  x_var='YrMo_Sold', 
                                                  y_var='Mean_Price', 
                                                  line_color=all_year_line_color,
                                                  yaxis_title='Monthly Mean Price')
    # bar plot: call function create_bar_plot()
    bar_plot_allyears = create_bar_plot(df=mp_df_year, y_var='Mean_Price',
                                        yaxis_title='Annual Mean Price',
                                        hover_temp='Year: %{x}<br>Annual Mean Price: %{y}')
    # histogram: sale price all data points
    hist_sp_allyears = create_histogram(df=ames, x_var='Sale_Price')
    
    # histogram: sale price of each year
    hist_sp_sepyears = create_ridgeline_plot(df=ames, 
                                             hl_year=None, x_var='Sale_Price', 
                                             year_lst=year_lst)
    
    ### initialize all label and graphs in the card
    if n == 0:
        return date_label_allyears, mp_indic_allyears, line_plot_allyears, bar_plot_allyears, hist_sp_allyears, hist_sp_sepyears
    ### update all label and graphs in the card with button
    elif 'button-visualize-sale-price' == ctx.triggered_id:
        date_label, mp_indic, line_plot, bar_plot, hist_box_plot, hist_plot = '', {}, {}, {}, {}, {}
        # graphs for all years
        if year == 'All':
            date_label = date_label_allyears
            mp_indic = mp_indic_allyears
            line_plot = line_plot_allyears
            bar_plot = bar_plot_allyears
            hist_box_plot = hist_sp_allyears
            hist_plot = hist_sp_sepyears
        
        # graphs for a specific year
        elif month == 'All':
            # year-month-label:
            date_label = ('{}.01 ~ {}.12'.format(year, year) if year != '2010' 
                          else '2010.01 ~ 2010.07')
            # mean price indicator
            indic_val = mp_df_year[mp_df_year['Year_Sold'] == int(year)]['Mean_Price'].values[0]
            ref_val = 0
            if year == '2006':
                ref_val = indic_val
            else:
                ref_val = mp_df_year[mp_df_year['Year_Sold'] == int(year) - 1].Mean_Price.values[0]
            
            suffix_text = ' vs. previous year'
            
            mp_indic = create_indicator(val=indic_val, 
                                        ref_val=ref_val, 
                                        delta_suffix=suffix_text, 
                                        fig_width=500, 
                                        num_prefix='$')
            # line plot: a line for each year
            line_plot = create_lineplot_sepyears(df=mp_df_month, 
                                                 x_var='month', y_var='Mean_Price', 
                                                 hl_year=year,
                                                 hl_month=None, year_lst=year_lst,
                                                 year_color_dict=year_color_dict,
                                                 year_hl_color_dict=year_hl_color_dict,
                                                 yaxis_title='Monthly Mean Price')
            
            # bar graph: all years, highlight the selected year
            color_seq = [year_hl_color_dict[y] if y == year 
                         else year_color_dict[y] for y in year_lst]
            bar_plot = create_bar_plot(df=mp_df_year, y_var='Mean_Price', 
                                       color_sequence=color_seq,
                                       yaxis_title='Annual Mean Price',
                                       hover_temp='Year: %{x}<br>Annual Mean Price: %{y}')
            # hist/ box plot: bottom left
            hist_box_plot = create_box_plot(df=ames[ames['Year_Sold'] == int(year)], 
                                            year=year,
                                            hl_month=None)
            hist_plot = create_ridgeline_plot(df=ames,
                                              hl_year=year, x_var='Sale_Price', 
                                              year_lst=year_lst)
        # graph for a specific month
        else:
            date_label = year + '.' + month
            # volume indicator
            indic_val = mp_df_month[(mp_df_month['year'] == int(year))
                                    & (mp_df_month['month'] == int(month))]['Mean_Price'].values[0]
            ref_val = 0
            if month == '01':
                ref_val = indic_val
            else:
                ref_val = mp_df_month[(mp_df_month['year'] == int(year))
                                      & (mp_df_month['month'] == int(month)-1)]['Mean_Price'].values[0]
            
            suffix_text = ' vs. previous month'
            mp_indic = create_indicator(val=indic_val, 
                                        ref_val=ref_val, 
                                        delta_suffix=suffix_text, 
                                        fig_width=400, 
                                        num_prefix='$')
            # line plot: a line for each year, highlight a point
            line_plot = create_lineplot_sepyears(df=mp_df_month, 
                                                 x_var='month', y_var='Mean_Price', 
                                                 hl_year=year,
                                                 hl_month=month, year_lst=year_lst,
                                                 year_color_dict=year_color_dict,
                                                 year_hl_color_dict=year_hl_color_dict,
                                                 yaxis_title='Monthly Mean Price')
            # bar graph: all years, highlight the selected year
            color_seq = [year_hl_color_dict[y] if y == year 
                         else year_color_dict[y] for y in year_lst]
            bar_plot = create_bar_plot(df=mp_df_year, y_var='Mean_Price',
                                       color_sequence=color_seq,
                                       yaxis_title='Annual Mean Price',
                                       hover_temp='Year: %{x}<br>Annual Mean Price: %{y}')
            
            # hist/ box plot: bottom left
            hist_box_plot = create_box_plot(df=ames[ames['Year_Sold'] == int(year)], 
                                            year=year,
                                            hl_month=month)
            hist_plot = create_ridgeline_plot(df=ames,
                                              hl_year=year, x_var='Sale_Price', 
                                              year_lst=year_lst)
        
        return date_label, mp_indic, line_plot, bar_plot, hist_box_plot, hist_plot
    else:
        return dash.no_update







