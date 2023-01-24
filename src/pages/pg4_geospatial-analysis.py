# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 19:06:11 2022

@author: ZNL
"""

import dash
from dash import dcc, html, callback, Output, Input, State, ctx
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import dash_bootstrap_components as dbc
import pandas as pd
import pathlib
import json
from sklearn.preprocessing import OrdinalEncoder

dash.register_page(__name__,
                   path='/geospatial-analysis',  # '/' is home page and it represents the url
                   name='Geospatial Analysis',  # name of page, commonly used as name of link
                   title='geospatial-analysis',  # title that appears on browser's tab
                   # image='pg1.png',  # image in the assets folder
                   description='geospatial feature analysis.'
)

# =============================================================================
# Load dataset, geojson files, data wrangling and preparation, pre-defined lists/dicts
# =============================================================================
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath('data').resolve()

ames = pd.read_csv(DATA_PATH.joinpath('ames_new.csv'))
ames = ames[['Sale_Price'] + [col for col in ames.columns if col != 'Sale_Price']]

with open(DATA_PATH.joinpath('ord_feat_levels_update.json'), 'r') as f:
    ord_feat_levels = json.load(f)

with open(DATA_PATH.joinpath('var_type_dict.json'), 'r') as f:
    var_type_dict = json.load(f)
    
with open(DATA_PATH.joinpath('var_desc_dict.json'), 'r') as f:
    var_desc_dict = json.load(f)

with open(DATA_PATH.joinpath('var_desc_dict_full.json'), 'r') as f:
    var_desc_dict_full = json.load(f)

ames_schools = pd.read_csv(DATA_PATH.joinpath('ames_schools_geo.csv'))

with open(DATA_PATH.joinpath('ames_school_dist_geojson.geojson'), 'r') as f:
    ames_school_geojson = json.load(f)

ames_school_dist = gpd.read_file(DATA_PATH.joinpath('ames_school_dist_geojson.geojson'))
    
# convert dtype of ordinal features to pandas Categorical type using ord_feat_levels
ames[var_type_dict['ordinal']] = (ames[var_type_dict['ordinal']]
                   .apply(lambda col: pd.Categorical(col, categories=ord_feat_levels[col.name], ordered=True)))

# add ordinal encoded features for all ordinal features
ord_feats_encode = ['ord_' + feat for feat in var_type_dict['ordinal']]
ord_enc = OrdinalEncoder(categories = [ames[feat].cat.categories.tolist() for feat in var_type_dict['ordinal']])
ames[ord_feats_encode] = ord_enc.fit_transform(ames[var_type_dict['ordinal']]) + 1

# define new variable type mapping dictionary
var_type_dict_new = {}
var_type_dict_new['geo_related'] = ['MS_Zoning', 'Neighborhood', 'Condition_1']
for typ in ['continuous', 'discrete', 'ordinal']:
    var_type_dict_new[typ] = var_type_dict[typ]
var_type_dict_new['nominal'] = [var for var in var_type_dict['nominal'] if var not in var_type_dict_new['geo_related']]

var_types_new = list(var_type_dict_new.keys()) # list of variable types used in this page

# =============================================================================
# Create all Components for layout
# =============================================================================
### define functions
# popover for highlight data checklist
def make_popover(body_text, target, header='**When Checked**'):
    popover = dbc.Popover([
        dbc.PopoverHeader([
            dcc.Markdown([header])
            ],
            class_name='py-1'),
        dbc.PopoverBody([
            dcc.Markdown([
                body_text
                ])
            ],
            class_name='p-1')
        ],
        target=target,
        trigger='hover',
        placement='right')
    return popover

### basic map graph
base_map = go.Figure()
# choroplethmap of school districts
base_map.add_trace(
    go.Choroplethmapbox(geojson=ames_school_geojson, 
                        locations=ames_school_dist.district, featureidkey='properties.district',
                        z=ames_school_dist.index, 
                        colorscale=px.colors.qualitative.Set3, showscale=False, # px.colors.qualitative.Set3, # zmin=0, zmax=4,
                        marker_opacity=0.5, marker_line_width=1.5, marker_line_color='#0074D9',
                        name='school<br>district', 
                        visible=True))
# scatter plot of school locations
base_map.add_trace(
    go.Scattermapbox(lat=ames_schools['Latitude'][:5], lon=ames_schools['Longitude'][:5], 
                     mode='markers', 
                     marker=go.scattermapbox.Marker(size=12, opacity=1, color='red'),
#                                                     color=ames_schools.index[:5], 
#                                                     colorscale=px.colors.qualitative.Set3, 
#                                                     showscale=False), 
                     text=ames_schools['School'][:5],
                     hovertemplate='<b>%{text}</b><br>Location: (%{lat}, %{lon})',
#                      hoverinfo='text',
                     name='School Location',
                     visible=True))
# initialize map layout
base_map.update_layout(
    # uirevision= 'foo',
    mapbox=dict(style='open-street-map', # open-street-map, carto-positron, stamen-watercolor, light(token)
#                 accesstoken=token,
                zoom=12,
                center={'lat':42.025, 'lon':-93.635}),
    hovermode='closest',
    title='<b>House Sale Distribrtion</b> | Ames, Iowa 2006-2010',
    legend=dict(orientation='h', y=-0.02, yanchor='top'), # , entrywidth=0
    margin=dict(l=0, r=0, t=30, b=0), 
    height=850, 
    # width=800
    ) # legend=dict(orientation='h')

### layout components
card_map_sidebar = dbc.Card([
    dbc.CardBody([
        dbc.Button(['Customize Visualization'], 
                   active=False, 
                   class_name='fw-bold fs-6 text-white bg-warning py-2 px-2 mx-1 ms-5 text-center'),
        html.Div([
            dbc.Label('data point colored by variable:', width='auto', 
                      class_name='text-info fw-bold mb-0')
            ],
            className='ms-1 mb-0 p-0'),
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label('Type', width='auto', 
                              class_name='text-info fw-bold d-inline me-0',
                              style={'font-size': '14px'})
                    ],
                    width={'size': 'auto', 'offset': 1},
                    class_name='me-0'),
                dbc.Col([
                    dbc.Select(id='dp-var-type',
                               options=[
                                   {'label': val, 'value': val}
                                   for val in var_types_new
                                   ],
                               value='geo_related',
                               class_name='p-1 text-info d-inline ms-0',
                               style={'font-size': '15px'}
                               ),
                    make_popover(
                        body_text='''- Type **geo_related** contains vairables with 
                                    geospatial information\n- geo_related variables 
                                    are originally included in nominal type\n- **Ordinal 
                                    variables are encoded** with lowest level as 1''', 
                                 target='dp-var-type', 
                                 header='**About Type**')
                    ],
                    class_name='ms-0',
                    width=8)
                ])
            ],
            className='mb-1 mt-0 ms-1 p-0'),
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label('Name',
                              width='auto',
                              class_name='text-info fw-bold me-0 d-inline',
                              style={'font-size': '14px'})
                    ],
                    width={'size': 'auto', 'offset': 1},
                    class_name='me-0'),
                dbc.Col([
                    dbc.Select(id='dp-var-name',
                               options=[],
                               required=True,
                               class_name='p-1 text-info d-inline ms-0',
                               style={'font-size': '14px'}
                               )
                    ],
                    class_name='ms-0',
                    width=7)
                ]),
            ],
            className='ms-1'),
        html.Div([
            # check list show neighborhood mean and school district
            dbc.Checklist(
                id='cl-schooldist', 
                options=[
                    # {'label':'Use neighborhood mean', 'value':'nb-mean',
                    #  'input_id':'cl-nb-mean',
                    #  'disabled':True},
                    {'label':'Show school district', 'value':'school-dist'}
                ],
                value=['school-dist'],
                switch=True,
                class_name='ms-5 mt-2',
                label_checked_class_name='fw-bold text-info',
                input_checked_class_name='bg-primary',
                label_style={'font-size': '14px'}),
            # make_popover(
            #     body_text='''- each data point is colored by **neighborhood average** 
            #                 of the above selected variable\n- a **density color map** 
            #                 will be shown\n- **disabled** when geo_related or nominal variable is 
            #                 selected''', 
            #     target='cl-nb-mean'),
            make_popover(
                body_text='''- school districts will be shown in the map''', 
                target='cl-schooldist')
            ]),
        html.Div([
            # dbc.Select map type: scatter or desity plot
            dbc.Label('Select map type:', 
                      class_name='fw-bold text-info mt-3 mb-1',
                      # style={'font-size': '14px'}
                      ),
            dbc.RadioItems(
                id='ri-map-type',
                options=[
                    {'label':'scatter plot', 'value':'scatter-plot',
                     'input_id':'ri-scatter-plot'},
                    {'label':'density heatmap', 'value':'density-heatmap',
                     'input_id':'ri-density'}
                    ],
                value='scatter-plot',
                class_name='ms-5 mt-0',
                label_checked_class_name='fw-bold text-info',
                input_checked_class_name='border border-primary bg-primary',
                # label_style={'font-size': '14px'}
                ),
            make_popover(
                body_text='''- display **details** of each individual data point''',
                target='ri-scatter-plot',
                header='scatter plot'),
            make_popover(
                body_text='''- generally display **regional trends** to get an overall 
                            picture\n- **disabled** when a geo_related or nominal 
                            variable is selected''',
                target='ri-density',
                header='density heatmap')
            ],
            className='ms-1'),
        html.Div([
            # dbc.Select map style
            dbc.Label('Select map style:', 
                      class_name='fw-bold text-info mt-3 mb-1',
                      # style={'font-size': '14px'}
                      ),
            dbc.RadioItems(
                id='ri-map-style',
                options=[
                    {'label':'open street map', 'value':'open-street-map'},
                    {'label':'carto positron', 'value':'carto-positron'}
                    ],
                value='open-street-map',
                class_name='ms-5 mt-0',
                label_checked_class_name='fw-bold text-info',
                input_checked_class_name='border border-primary bg-primary',
                # label_style={'font-size': '14px'}
                )
            ],
            className='ms-1'),
        html.Div([
            html.H2(html.I(className=' bi bi-arrow-down-square',
                           style={'color': '#FC6955'}),
                    className='text-center fw-bold')
            ],
            className='mt-3'),
        html.Div([
            dbc.Button(['Visualize'], 
                       id='button-visualize',
                       active=True,
                       class_name='fw-bold text-white py-2 px-3 mt-0',
                       n_clicks=0,
                       style={'backgroundColor':'#FC6955'}) # FF9900
            ],
            className='text-center')
        ],
        class_name='py-2 px-1')
    ],
    class_name='border-info border-1 mt-1 p-2 mx-0 bg-light',
    style={'height':'60rem'})

card_map_graph = dbc.Card([
    dbc.CardHeader([
        dbc.Row([
            dbc.Label('colored by Neighborhood', id='header-varname-for-color',
                      class_name='text-info fw-bold')
            ],
            class_name='py-0')
        ],
        class_name='py-0'),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='map', figure={}, 
                          className='p-0 h-100')
                ],
                class_name='overflow-auto')
            ])
        ],
        class_name='p-1 overflow-auto')
    ],
    class_name='mt-1 mb-1 p-1',
    style={'height':'60rem'})

# =============================================================================
# Layout
# =============================================================================
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3('Geospatial Data Analysis', 
                    className='fw-bold my-0', 
                    style={'display':'inline-block'})
            ],
            class_name='fw-bold text-center mt-3 mb-1 border-2 border-bottom shadow-sm',
            width={'size': 6, 'offset': 1}),
        dbc.Col([
            dbc.Button(['About this Page'],
                       id='button-about-pg-4',
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
                    '''This page provides a simple interactive map layout with a sidebar for customized visualization.
                      It maps every sale record in the dataset with their geographic location and uses color to visually
                      represent information of interest. The color attribute of this scatter map can not only help display
                      geo-related features such as neighborhoods, house zoning, school district where each house is located, 
                      but also help visualize geographical distribution of ALL variables in the dataset. Density heatmap can be
                      also visualized with continuous and ordinal variables.'''
                    ]),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Img(src='assets/Picture13.png',
                                     height='150px',
                                     className='')
                            ],
                            width=3)
                        ],
                        class_name='justify-content-around')
                    ])
                ]),
            dbc.ModalFooter(
                html.P('Part 3: Geospatial Data Analysis'))
            ],
            id='modal-about-pg-4',
            size='lg',
            scrollable=True,
            is_open=False)
        ],
        justify='center'),
    # map
    dbc.Row([
        dbc.Col([
            card_map_sidebar
            ],
            width=3),
        dbc.Col([
            card_map_graph
            ],
            width=9)
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
    Output('modal-about-pg-4', 'is_open'),
    [Input('button-about-pg-4', 'n_clicks')],
    [State('modal-about-pg-4', 'is_open')]
)
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

# variable name dropdown options controled by variable type dropdown
# disable or enable 'use neighborhood mean' (not used)
@callback(
    Output('dp-var-name', 'options'),
    Output('ri-map-type', 'options'),
    Output('ri-map-type', 'value'),
    Input('dp-var-type', 'value'),
    State('ri-map-type', 'value')
    )
def toggle_var_name_map_type_options(var_typ, current_map_typ):
    var_name_options = [
        {'label': var, 'value': var}
        for var in var_type_dict_new[var_typ]
        ]
    
    options_1 = [
        {'label':'scatter plot', 'value':'scatter-plot',
         'input_id':'ri-scatter-plot'},
        {'label':'density heatmap', 'value':'density-heatmap',
         'input_id':'ri-density'}
        ]
    
    options_2 =[
        {'label':'scatter plot', 'value':'scatter-plot',
         'input_id':'ri-scatter-plot'},
        {'label':'density heatmap', 'value':'density-heatmap',
         'input_id':'ri-density',
         'disabled':True}
        ]
    
    map_type_options = (options_2 if var_typ in ['geo_related', 'nominal'] 
                        else options_1)
    map_typ = ('scatter-plot' if var_typ in ['geo_related', 'nominal'] 
               else current_map_typ) 
    
    # options_1 = [
    #     {'label':'Use neighborhood mean', 'value':'nb-mean',
    #      'input_id':'cl-nb-mean',
    #      'disabled':True},
    #     {'label':'Show school district', 'value':'school-dist',
    #      'input_id':'cl-school-dist'}
    # ]
    # options_2 = [
    #     {'label':'Use neighborhood mean', 'value':'nb-mean',
    #      'input_id':'cl-nb-mean',
    #      'disabled':False},
    #     {'label':'Show school district', 'value':'school-dist',
    #      'input_id':'cl-school-dist'}
    # ]
    # nb_schooldist_options = (options_1 if var_typ in ['geo_related', 'nominal'] 
    #                          else options_2)
    return var_name_options, map_type_options, map_typ

# variable name dropdown default value controled by options
@callback(
    Output('dp-var-name', 'value'),
    Input('dp-var-name', 'options'),
    )
def initialize_var_name_value(options):
    # value = ('Neighborhood' if options[0]['value'] in var_type_dict_new['geo_related'] 
    #          else options[0]['value'])
    if options[0]['value'] in var_type_dict_new['geo_related']:
        value = 'Neighborhood'
    elif options[0]['value'] in var_type_dict_new['continuous']:
        value = 'Sale_Price'
    elif options[0]['value'] in var_type_dict_new['ordinal']:
        value = 'Overall_Qual'
    else:
        value = options[0]['value']
    
    return value

# map card header, graph
@callback(
    Output('header-varname-for-color', 'children'),
    Output('map', 'figure'),
    Input('dp-var-type', 'value'),
    Input('dp-var-name', 'value'),
    Input('cl-schooldist', 'value'),
    Input('ri-map-type', 'value'), 
    Input('ri-map-style', 'value'),
    Input('button-visualize', 'n_clicks')
    )
def update_card_map(var_typ, var_name, sch_dist, map_typ, map_style, n):
    colors_nom = px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet[:4]
    map_output = go.Figure(base_map)
    if n == 0:
        # map_output = go.Figure(base_map)
        ### scatter plot of all data points colored by neighborhood
        # one trace for each neighborhood to correctly show legend
        for i, val in enumerate(ames.Neighborhood.value_counts().index):
            sub_df = ames[ames.Neighborhood == val]
            name = 'South_and_West_of_<br>Iowa_State_University' if val == 'South_and_West_of_Iowa_State_University' else val
        #     legendwidth = 250 if val in ['Iowa_DOT_and_Rail_Road', 'South_and_West_of_Iowa_State_University'] else 110
            map_output.add_trace(
                go.Scattermapbox(
                    lat=sub_df['Latitude'], lon=sub_df['Longitude'], 
                    mode='markers', 
#                          marker=dict(size=6, opacity=0.6,  color=cols_neighbor[i]),
                    marker=go.scattermapbox.Marker(size=6, opacity=0.6, 
                                                   color=colors_nom[i]), 
                                                    # colorscale=px.colors.qualitative.Dark24,# showscale=False
                    text=sub_df.Sale_Price,
                    hovertemplate='Sale Price: %{text}<br>Location: (%{lat}, %{lon})',
                    legendwidth = 150,
                    showlegend=True, 
#                      text=ames_schools['School'][:5], 
#                      hoverinfo='text', 
                    name=name, 
                    visible=True))
        header = 'colored by Neighborhood'
        return header, map_output
    
    elif 'button-visualize' == ctx.triggered_id:
        # map_output = go.Figure(base_map)
        if var_typ in ['continuous', 'discrete', 'ordinal']:
            var_name_new = 'ord_' + var_name if var_typ == 'ordinal' else var_name 
            custom_df = ames[[var_name_new] + ['Neighborhood', 'Sale_Price']]
            if var_name == 'Sale_Price':
                template = ('House Location: (%{lat}, %{lon})<br>Neighborhood: %{customdata[1]}<br>' 
                               + 'Sale_Price: %{customdata[2]}')
            else:
                template = ('%{text}: %{customdata[0]}<br>'
                               + 'House Location: (%{lat}, %{lon})<br>Neighborhood: %{customdata[1]}<br>' 
                               + 'Sale_Price: %{customdata[2]}')
            if map_typ == 'scatter-plot':
                map_output.add_trace(
                    go.Scattermapbox(
                        lat=ames['Latitude'], lon=ames['Longitude'],
                        mode='markers', 
                        marker=go.scattermapbox.Marker(
                            size=7, opacity=0.6,
                            color=ames[var_name_new], 
                            colorscale='Inferno', # Inferno, thermal, Magma, Plasma
                            cmin=ames[var_name_new].min(), 
                            cmax=ames[var_name_new].max(),
                            colorbar=dict(title={'text': var_name_new},
                                          outlinewidth=0,
                                          thickness=15),
                            showscale=True), 
                        text=[var_name_new] * ames.shape[0],
                        customdata=custom_df,
    #                      hoverlabel='Overall_Qual',
                        hovertemplate=template,
                        name=var_name_new, # remove prefix
                        visible=True))
            else: # map_typ == 'density-heatmap'
                map_output.add_trace(
                    go.Densitymapbox(
                        lat=ames['Latitude'], lon=ames['Longitude'], 
                        z=ames[var_name_new],
                        zmin=ames[var_name_new].min(), 
                        zmax=ames[var_name_new].max(),
                        colorscale='Plasma', # Inferno
                        colorbar=dict(title={'text': var_name_new},
                                      outlinewidth=0,
                                      thickness=15),
                        radius=10,
                        text=[var_name_new] * ames.shape[0],
                        customdata=custom_df,
   #                      hoverlabel='Overall_Qual',
                        hovertemplate=template,
                        name=var_name_new))
        
        else: # var_typ in ['geo_related', 'nominal']
            ### scatter plot of all data points colored by the variable
            # one trace for each category to correctly show legend
            legend_width = 250 if var_name == 'MS_SubClass' else 150
            for i, val in enumerate(ames[var_name].value_counts().index):
                sub_df = ames[ames[var_name] == val]
                name = 'South_and_West_of_<br>Iowa_State_University' if val == 'South_and_West_of_Iowa_State_University' else val
                map_output.add_trace(
                    go.Scattermapbox(
                        lat=sub_df['Latitude'], lon=sub_df['Longitude'], 
                        mode='markers', 
    #                          marker=dict(size=6, opacity=0.6,  color=cols_neighbor[i]),
                        marker=go.scattermapbox.Marker(
                            size=6, opacity=0.6, 
                            color=colors_nom[i]), 
                             # colorscale=px.colors.qualitative.Dark24,# showscale=False
                        text=sub_df.Sale_Price,
                        hovertemplate='Sale Price: %{text}<br>Location: (%{lat}, %{lon})',
                        legendwidth = legend_width,
                        showlegend=True, 
    #                      text=ames_schools['School'][:5], 
    #                      hoverinfo='text', 
                        name=name, 
                        visible=True))
        # update layout based on map style
        map_output.update_layout(
            mapbox=dict(style=map_style, # open-street-map, carto-positron, stamen-watercolor, light(token)
#                 accesstoken=token,
                zoom=12,
                center={'lat':42.025, 'lon':-93.635}))
        # make school district invisible based on checklist value
        if 'school-dist' not in sch_dist:
            map_output.data[0].visible, map_output.data[1].visible = False, False
        # define header
        header = 'colored by {}'.format(var_name)
        return header, map_output
    
    else:
        return dash.no_update
        


















