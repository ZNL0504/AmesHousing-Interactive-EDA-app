# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 08:31:44 2022

@author: ZNL
"""
import dash
from dash import dcc, html, callback, Output, Input, State
import plotly.express as px
import dash_bootstrap_components as dbc

dash.register_page(__name__,
                   path='/',  # '/' is home page and it represents the url
                   name='Home',  # name of page, commonly used as name of link
                   title='Dashboard-Outline',  # title that appears on browser's tab
                   # image='pg1.png',  # image in the assets folder
                   description='House Sale Evaluation Interactive Dashboard outline.'
)

# =============================================================================
# Layout
# =============================================================================
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2('Dashboard Outline and Resource Guide', 
                    className='fw-bold text-center my-3 border-2 border-bottom shadow-sm')
            ],
            width=10)], justify='center'),
    dbc.Row([
        dbc.Col([
            dbc.Carousel(
                items=[
                    {'key': '1', 'src': 'assets/Picture5_1.png', 
                     'header': 'Dashboard Outline', 
                     'img_style': {'max-height': '400px'}},
                    {'key': '2', 'src': 'assets/Picture6_2.png', 
                     'header': 'House Sale Overview', # Dataset Exploration and Analysis
                     'img_style': {'max-height': '400px'}},
                    {'key': '3', 'src': 'assets/Picture7_1.png', 
                     'header': 'Dataset Exploration and Statistical Inference', # Geospatial Feature Analysis
                     'img_style': {'max-height': '400px'}},
                    {'key': '4', 'src': 'assets/Picture8.png', 
                     'header': 'Geospatial Data Analysis',
                     'img_style': {'max-height': '400px'}}
                ],
                controls=True,
                indicators=True,
                interval=2500,
                ride='carousel',
                variant='dark',
                class_name='bg-gradient', # carousel-fade
                style={
                    'backgroundColor': 'rgb(179, 205, 227)'# 204, 230
                    })
            ], 
            width=10)], justify='center'),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            dbc.Button('About Dataset', id='about-dataset-button', 
                       n_clicks=0, active=True,
                       # color='primary',
                        class_name='bg-primary text-white fw-bold', 
                       # style={'border': '2px solid darkblue'}
                       ),
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle('About Dataset')),
                dbc.ModalBody([
                    html.Div([
                        html.Img(src='assets/Picture9_2.png',
                                 height='160px',
                                 className='ps-0 ms-0 mb-2')]),
                    dcc.Markdown([
                        '''This interactive data visualization and analysis dashboard will be utilizing dataset from 
                          **[AmesHousing R Package](https://cran.r-project.org/web/packages/AmesHousing/index.html)**.
                          \nThis package not only includes original ames dataset same as the Kaggle competition dataset 
                          **[House Prices - Advanced Regression Technique](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)**,
                          but also includes a *processed* version of the original Kaggle dataset with all missing data imputated, and most importantly, it provides 
                          additional geographic information (approximate longitude and latitude) for each sale record, which can be utilized to reveal how geographic 
                          locations affect house sale prices.
                          \nThe dataset used in this dashboard has been further processed, which mainly involves new feature generation and outlier data removal. Refer to 
                          **[AmesHousing_R_Package_EDA_with_plotly](https://github.com/ZNL0504/AmesHousing-Interactive-EDA-app/blob/main/dataset_download_and_EDA/AmesHousing_R_Package_EDA_with_Plotly.ipynb)** (.ipynb file) for full details.
                          \nIf you are interested in dataset downloading from scratch, refer to file **[AmesHousing_dataset.Rmd](https://github.com/ZNL0504/AmesHousing-Interactive-EDA-app/blob/main/dataset_download_and_EDA/AmesHousing_dataset_download.Rmd)** (R markdown file), where details can be found for how to import this package in
                          RStudio/R and save it as .csv file.
                          \n**Package Reference:**
                          \n- [package reference manual](https://cran.r-project.org/web/packages/AmesHousing/AmesHousing.pdf)\n- **Source:**  \nDe Cock, D. (2011). "Ames, Iowa: Alternative to 
                          the Boston Housing Data as an End of Semester Regression Project," Journal of Statistics Education, Volume 19, Number 3.\n- [link to the article](http://ww2.amstat.org/publications/jse/v19n3/decock.pdf)\n- [data documentation](https://ww2.amstat.org/publications/jse/v19n3/decock/DataDocumentation.txt)'''
                        ])
                    ]),
                dbc.ModalFooter(
                    html.P('House Sale Evaluation'))
                ],
                id='modal-about-dataset',
                size='lg',
                scrollable=True,
                is_open=False)
            ], 
            width={'size': 3, 'offset': 1}),
        dbc.Col([
            dbc.Button('About Dashboard', id='workflow-button',
                       n_clicks=0, active=True,
                       class_name='bg-primary text-white fw-bold',
                       # style={'border': '2px solid darkblue'}
                       ),
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle('About Dashboard')),
                dbc.ModalBody([
                    dcc.Markdown([
                        '''This dashboard is an extension of data science project: **[House Price Prediction – Comprehensive Data Science Project](https://github.com/ZNL0504/House-Price-Prediction)**.
                          The goal of this dashboard is to interactively visualize data, interpret analytical results, and extract data insights in a very flexible and intuitive approach. 
                          \nThe main content of the dashboard is divided into three parts, each with one page.\n- The first part is house sale overview, where trend and pattern over time is shown in terms of sale volumn, gross 
                          sale and mean sale price.\n- The second part focuses on dataset exploration and statistical inference to provide evidence of most important house price predictors and reveal relationship among different
                          variables.\n- It ends with geospatial data analysis to show how locations can impact house sale price and relate to other features in the dataset.
                          \nIn summary, the project provides a data story telling platform, and the dashboard help bring the story to life.
                          The image below depicts the outline and workflow of the whole project, as well as the basic components of the dashboard structure.'''
                        ]),
                    html.H5('Project Outline',
                            className='text-left fw-bold'),
                    dbc.Row([
                        html.Img(key='workflow',
                                 src='assets/Picture10_3.png', 
                                 width='400px'
                                 )], 
                        justify='center')
                               ]),
                dbc.ModalFooter(
                    html.P('House Sale Evaluation'))
                ],
                id='modal-project-workflow',
                size='xl',
                scrollable=True,
                is_open=False)
            ], 
            width=3),
        dbc.Col([
            dbc.Button('More Resources', id='resource-button',
                       n_clicks=0, active=True,
                       class_name='bg-primary text-white fw-bold',
                       # style={'border': '2px solid darkblue'}
                       ),
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle('More Resources')),
                dbc.ModalBody([
                    dcc.Markdown([
                        '''Below are some very useful online resources that help with completing this dashboard.
                          \n- **[Charming Data](https://www.youtube.com/@CharmingData)** - a youtube channel for learning dash
                          plotly.\n- **[datacamp online course - Building Dashboards with Dash and Plotly](https://app.datacamp.com/learn/courses/building-dashboards-with-dash-and-plotly)**\n- **[Plotly Open Source Graphing Library
                          for Python](https://plotly.com/python/)**\n- **[Dash Python User Guide](https://dash.plotly.com/)**\n- **[Dash Enterprise App Gallery](https://dash.gallery/Portal/)**'''
                        ]),
                    ]),
                dbc.ModalFooter(
                    html.P('House Sale Evaluation'))
                ],
                id='modal-more-resources',
                size='lg',
                scrollable=True,
                is_open=False)
            ], 
            width=3)
        ], 
        justify='center',
        className='my-3 border-2 pb-5 shadow-sm')
    ],
    fluid=True
    )

# =============================================================================
# Callbacks
# =============================================================================
# home page modal 1
@callback(
    Output('modal-about-dataset', 'is_open'),
    [Input('about-dataset-button', 'n_clicks')],
    [State('modal-about-dataset', 'is_open')]
)
def toggle_modal_1(n, is_open):
    if n:
        return not is_open
    return is_open

# home page modal 2
@callback(
    Output('modal-project-workflow', 'is_open'),
    [Input('workflow-button', 'n_clicks')],
    [State('modal-project-workflow', 'is_open')]
)
def toggle_modal_2(n, is_open):
    if n:
        return not is_open
    return is_open

# home page modal 3
@callback(
    Output('modal-more-resources', 'is_open'),
    [Input('resource-button', 'n_clicks')],
    [State('modal-more-resources', 'is_open')]
)
def toggle_modal_3(n, is_open):
    if n:
        return not is_open
    return is_open


















