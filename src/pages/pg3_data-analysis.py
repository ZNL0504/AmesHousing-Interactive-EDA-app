# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 08:41:16 2022

@author: ZNL
"""
import dash
from dash import dcc, html, callback, Output, Input, State, ctx
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable, FormatTemplate
import pandas as pd
import pathlib
import json
import re
from stats_inference_func import stats_inference

dash.register_page(__name__,
                   path='/data-analysis',  # '/' is home page and it represents the url
                   name='Data Analysis',  # name of page, commonly used as name of link
                   title='data-analysis',  # title that appears on browser's tab
                   # image='pg1.png',  # image in the assets folder
                   description='Dataset exploration and analysis.'
)

# =============================================================================
# Load dataset, data wrangling and preparation, pre-defined lists/dicts
# =============================================================================
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath('data').resolve()

ames = pd.read_csv(DATA_PATH.joinpath('ames_new.csv'))
ames = ames[['Sale_Price'] + [col for col in ames.columns if col != 'Sale_Price']]

feat_coef_df = pd.read_csv(DATA_PATH.joinpath('feat_coef_df.csv'))
feat_coef_df_new = feat_coef_df.set_index('feature_v2', drop=True)

# features that need tooltip in data table
feats_tooltip = ['MS_SubClass', 'MS_Zoning', 'Lot_Shape', 'Neighborhood']

with open(DATA_PATH.joinpath('ord_feat_levels_update.json'), 'r') as f:
    ord_feat_levels = json.load(f)

with open(DATA_PATH.joinpath('var_type_dict.json'), 'r') as f:
    var_type_dict = json.load(f)
    
with open(DATA_PATH.joinpath('var_desc_dict.json'), 'r') as f:
    var_desc_dict = json.load(f)

with open(DATA_PATH.joinpath('var_desc_dict_full.json'), 'r') as f:
    var_desc_dict_full = json.load(f)
    
# convert dtype of ordinal features to pandas Categorical type using ord_feat_levels
ames[var_type_dict['ordinal']] = (ames[var_type_dict['ordinal']]
                   .apply(lambda col: pd.Categorical(col, categories=ord_feat_levels[col.name], ordered=True)))

# used as argument of statistical inference function
twolev_nom_vars = [feat for feat in var_type_dict['nominal'] if len(ames[feat].unique()) == 2]
twolev_ord_vars = [feat for feat in var_type_dict['ordinal'] if len(ames[feat].unique()) == 2]
twolev_cat_vars = twolev_nom_vars + twolev_ord_vars

var_types = list(var_type_dict.keys()) # list of variable types
dtype_colors = px.colors.qualitative.Dark2[:6]

dtype_color_map = dict() # dictionary for data type and color mapping
for typ, color in zip(var_types, dtype_colors):
    dtype_color_map[typ] = color

alpha = 0.1
dtype_color_map_rgba = dict()
for typ, color in dtype_color_map.items():
    dtype_color_map_rgba[typ] = 'rgba(' + re.findall('[0-9,]+', color)[0] + ',{})'.format(alpha)

# print(ames.Exter_Cond.dtype)
# =============================================================================
# Create all Components for layout
# =============================================================================
### define functions
# popover for highlight data checklist
def make_popover(typ, body_text, target):
    popover = dbc.Popover([
        dbc.PopoverHeader([
            dbc.Badge(typ, color=dtype_color_map[typ],
                      class_name='me-1 p-1')
            ],
            class_name='py-1'),
        dbc.PopoverBody([body_text],
                        class_name='p-1')
        ],
        target=target,
        trigger='hover',
        placement='right')
    
    return popover

# popover for feature importance visualization dbc.Select 
def make_popover_feat_impt(body_text, target, header):
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
        placement='top')
    return popover

# data bars in Sale_Price column
# source code from: https://dash.plotly.com/datatable/conditional-formatting
def data_bars(df, column):
    n_bins = 100
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ranges = [
        ((df[column].max() - df[column].min()) * i) + df[column].min()
        for i in bounds
    ]
    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        max_bound_percentage = bounds[i] * 100
        styles.append({
            'if': {
                'filter_query': (
                    '{{{column}}} >= {min_bound}' +
                    (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                'column_id': column
            },
            'background': (
                """
                    linear-gradient(90deg,
                    #0074D9 0%,
                    #0074D9 {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(max_bound_percentage=max_bound_percentage)
            ),
            'paddingBottom': 2,
            'paddingTop': 2
        })

    return styles

### layout components
# data table and single variable analysis header
card_first_part_header = dbc.Card([
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4('Dataset Overview',
                            className='fw-bold', 
                            style={'color': '#485785'}
                            ),
                    dcc.Markdown([
                        '''Let's start data exploration with an interactive data table where variables can be sorted, 
                           explained (with header tooltips), and highlighted based on data types (sidebar). Then use VARIABLE SELECT 
                           options right under the table to zoom in on single variable analysis, which can further help make these variables
                           more intuitive and meaningful.'''],
                           className='m-0 text-djustify',
                           style={'line-height': '1.2'})
                    ])
                ],
                width=5),
            # image
            dbc.Col([
                html.Div([
                    html.Img(src='assets/Picture0.png',
                             height='140px',
                             className='ps-0 ms-0')]),
                ],
                class_name='ps-0 pe-1 ms-0',
                width=7),
            ])
        ], 
         class_name='m-0 pb-0 pt-2 px-2')
    ],
     class_name='border-body border-2 border-bottom-0 border-start-0 border-end-0 pb-0 px-2 mt-2 bg-gradient shadow-lg')

card_datatable_sidebar = dbc.Card([
    dbc.CardBody([
        dbc.Button(['Highlight Data'], 
                   active=False, 
                   class_name='fw-bold fs-6 text-white bg-warning py-2 px-2 mt-0 mb-2'),
        html.Div([
            dbc.Label(['Numerical Data'], class_name='fw-bold mb-1'),
            # dbc.Switch(id='cb-num-var',
            #            label='continuous', value=False,
            #            label_style={'color': dtype_color_map['continuous']},
            #            # input_style={'backgroundColor': dtype_color_map['continuous']}
            #            ),
            dbc.Checklist(id='cl-num-var', 
                          options=[
                              {'label':'continuous', 'value':'continuous', 
                               'input_id':'cl-cont-var'},
                              {'label':'discrete', 'value':'discrete',
                               'input_id':'cl-discrete-var'}
                          ],
                          value=[],
                          switch=True,
                          class_name='ms-3 mt-0',
                          label_checked_class_name='fw-bold',
                          # label_checked_style={'color': dtype_colors[0]},
                          input_checked_class_name='bg-primary')
            ]),
        html.Div([
            dbc.Label(['Categorical Data'], class_name='fw-bold mb-1'),
            dbc.Checklist(id='cl-cat-var', 
                          options=[
                              {'label':'nominal', 'value':'nominal',
                               'input_id':'cl-nom-var'},
                              {'label':'ordinal', 'value':'ordinal',
                               'input_id':'cl-ord-var'}
                          ],
                          value=[],
                          switch=True,
                          class_name='ms-3 mt-0',
                          label_checked_class_name='fw-bold',
                          input_checked_class_name='bg-primary')
            ],
            className='mt-2 mb-2'),
        html.Div([
            dbc.Label(['Other Data'], class_name='fw-bold mb-1'),
            dbc.Checklist(id='cl-other-var', 
                          options=[
                              {'label':'time', 'value':'time',
                               'input_id':'cl-time-var'},
                              {'label':'geospatial', 'value':'geospatial',
                               'input_id':'cl-geo-var'}
                          ],
                          value=[],
                          switch=True,
                          class_name='ms-3 mt-0',
                          label_checked_class_name='fw-bold',
                          input_checked_class_name='bg-primary')
            ]),
        html.Div([
            html.H1(html.I(className=' bi bi-arrow-down-square text-warning fw-bold'))
            ],
            className='mt-3 ms-5'),
        make_popover(typ='continuous', 
                     body_text='numerical variable with continuous values',
                     target='cl-cont-var'),
        make_popover(typ='discrete', 
                     body_text='numerical variable with discrete values',
                     target='cl-discrete-var'),
        make_popover(typ='nominal', 
                     body_text='categorical variable without orders',
                     target='cl-nom-var'),
        make_popover(typ='ordinal', 
                     body_text='categorical variable with specific orders',
                     target='cl-ord-var'),
        make_popover(typ='time', 
                     body_text='time-related variables (year, month)',
                     target='cl-time-var'),
        make_popover(typ='geospatial', 
                     body_text='longitude and latitude variables',
                     target='cl-geo-var')
        ],
        class_name='py-2')
    ],
    class_name='border-info border-1 mt-1 p-0 mx-0 bg-light',
    style={'height':'24rem'})

card_datatable = dbc.Card([
    dbc.CardHeader([
        dbc.Row([
            html.P(['The Ames Iowa Housing Data'], 
                   className='d-inline text-info fw-bold'),
            html.Footer(['2006-2010'], className='small d-inline'),
            html.A(['Package Link'], 
                   href='https://cran.r-project.org/web/packages/AmesHousing/index.html',
                   className='small'),
            html.A(['Documentation'],
                   href='https://cran.r-project.org/web/packages/AmesHousing/AmesHousing.pdf',
                   className='small')
            ],
            class_name='d-inline py-0')
        ],
        class_name='pt-1 pb-0 border-bottom-0'),
    
    dbc.CardBody([
        DataTable(
            id='table',
            columns=([{'name': 'Sale_Price', 'id': 'Sale_Price', 
                       'type':'numeric', 'format':FormatTemplate.money(0)}] 
                     + [{'name': col, 'id': col, 
                         'type': 'text' if ames[col].dtype == 'object' else 'numeric'}
                        for col in ames.columns[1:]]),
            data=ames.to_dict('records'),
            editable=False,
            cell_selectable=False,
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
            
            style_cell={
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                # 'maxHeight': '6px',
                'minWidth': '130px', 'width': '130px', 'maxWidth': '130px',
                # 'whiteSpace': 'normal',
                'padding': '1px 8px',
                'font-size': '14px',
                },
            
            style_data={
                'font-size': '12px'
                },
            
            style_header={
                'backgroundColor':'rgb(240, 240, 240)',
                'fontWeight': 'bold',
                'textDecoration': 'underline',
                'textDecorationStyle': 'dotted'
                },
            
            style_table={
                'height': '330px',
                'minWidth': '100%',
                # 'overflowX':'scroll',
                # 'overflowY': 'scroll'
                },
            
            style_data_conditional=(
                [{
              'if': {'row_index': 'odd'},
              'backgroundColor': 'rgb(250, 250, 250)'
              }]
                +
                [
                {
                 'if': {'column_id': col},
                 'textDecoration': 'underline',
                 'textDecorationStyle': 'dotted'
                    } for col in feats_tooltip
                ]
                +
                data_bars(ames, 'Sale_Price')
                ),
            
            tooltip_header={
                col: {'value': ('**{}:**\n- {}\n- {}'
                                .format(col,
                                        var_desc_dict[col][0],
                                        var_desc_dict[col][1])),
                      'type': 'markdown'
                      } for col in ames.columns
                },
            
            tooltip_data=[
                {
                    col: {'value': str(val), 'type': 'markdown'}
                    for col, val in row.items()
                } for row in ames[feats_tooltip].to_dict('records')
                ],
            
            tooltip_delay=0,
            tooltip_duration=None,
            
            # style tooltip
            css=[{
                'selector': '.dash-table-tooltip',
                'rule': 'background-color: rgb(252, 252, 252); border: 1px solid #0074D9; font-size: 14px; padding: 2px 2px'
                }],
            
            fixed_columns={'headers': True, 'data': 1},
            fixed_rows={'headers': True, 'data': 0},
            virtualization=True,
            page_action='none'
            )
        ],
        class_name='p-1'),
    
    dbc.CardFooter([
        '{} rows x {} columns'.format(ames.shape[0], ames.shape[1])
        ],
        class_name='py-0 small')
    ],
    class_name='mt-1 mx-0')

form_feat_select = dbc.Form(
    dbc.Row(
        [
            dbc.Label('Select Variable Type', width='auto', 
                      class_name='text-info fw-bold'),
            dbc.Col(
                dbc.Select(id='dp-select-var-type-1',
                           options=[
                               {'label': val, 'value': val}
                               for val in var_types 
                               if val not in ['time', 'geospatial']
                               ],
                           value='continuous',
                           class_name='p-1 text-info',
                           style={'font-size': '16px'}
                           ),
                width=2,
                class_name="me-2",
            ),
            dbc.Col(
                html.H4(html.I(className=' bi bi-arrow-right-circle me-2 text-info')), # bi bi-caret-right-fill
                width='auto'
                ),
            dbc.Label('Variable Name', width='auto',
                      class_name='text-info fw-bold'),
            dbc.Col(
                dbc.Select(id='dp-select-var-name-1',
                           options=[],
                           required=True,
                           class_name='p-1 text-info',
                           style={'font-size': '14px'}
                           ),
                width=2,
                class_name="me-3",
            ),
            dbc.Col(
                html.H4(html.I(className=' bi bi-arrow-right-circle me-2 text-info')), # bi bi-caret-right-fill
                width='auto'
                ),
            dbc.Col(
                dbc.Button(['Visualize Data'], 
                           id='button-visualize-data',
                           active=True,
                           class_name='fw-bold text-white py-2 px-3 mx-1',
                           n_clicks=0,
                           style={'backgroundColor':'#FF9900'}), 
                width='auto'
                )
        ],
        class_name='g-2 mt-1 mb-1 align-items-center',
    )
)

card_var_graph_desc = dbc.Card([
    dbc.CardHeader([
        dbc.Row([
            dbc.Label('Sale_Price', id='header-var-name',
                      class_name='d-inline text-info fw-bold'),
            dbc.Badge('continuous', id='header-var-type',
                      color=dtype_color_map['continuous'],
                      class_name='mx-1 p-1 d-inline')
            ],
            class_name='d-inline')
        ],
        class_name='py-1'),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='fig-single-var', 
                          # initialize figure
                          figure={}, 
                          className='p-0 h-100')
                ],
                style={'height':'16rem'}, # adjust according to different variables
                class_name='p-0',
                width=6),
            dbc.Col([
                dcc.Markdown([],
                    id='var-desc',
                    className='text-info px-0 h-100 bg-transparent',
                    # style={'overflow': 'scroll'}
                    )
                ],
                # style={'height':'15rem'},
                class_name='h-100 px-0',
                width=5)
            ],
            justify='center',
            class_name='h-100 p-0')
        ],
        class_name='bg-transparent overflow-auto')
    ],
    class_name='mt-0 mb-1 p-1',
    style={'height':'20rem'})

# statistical inference header
card_2nd_part_header = dbc.Card([
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4('Statistical Inference',
                            className='fw-bold', 
                            style={'color': '#485785'}
                            ),
                    dcc.Markdown([
                        '''Let's further explore relationship among vairables. Select a pair of variables 
                           to VISUALIZE their relation and check CORRELATION/HYPOTHESIS 
                           TEST results. What features do you care about most in house sale? Is your intuition
                           consistent with the inference?'''],
                           className='m-0 text-djustify',
                           style={'line-height': '1.2'})
                    ])
                ],
                width=4),
            # image
            dbc.Col([
                html.Div([
                    html.Img(src='assets/Picture4.png',
                              height='150px',
                              className='ps-0 ms-0')]),
                ],
                class_name='ps-0 pe-1 ms-0',
                width=8),
            ])
        ], 
         class_name='m-0 pb-0 pt-2 ps-2 pe-1')
    ],
     class_name='border-body border-2 border-bottom-0 border-start-0 border-end-0 pb-0 px-2 mt-2 bg-gradient shadow-lg')

form_2feat_select = dbc.Form([
    dbc.Row([
        dbc.Col([
            dbc.Label('Select Variable 1: ', width='auto', 
                      class_name='text-info me-0 fw-bold d-inline')
            ],
            width='auto'),
        dbc.Col([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Type', width='auto', 
                                  class_name='text-info fw-bold d-inline',
                                  style={'font-size': '15px'})
                        ],
                        width='auto',
                        class_name='me-1'),
                    dbc.Col([
                        dbc.Select(id='dp-select-var-type-2',
                                   options=[
                                       {'label': val, 'value': val}
                                       for val in var_types 
                                       if val not in ['time', 'geospatial']
                                       ],
                                   value='continuous',
                                   class_name='p-1 text-info d-inline',
                                   style={'font-size': '16px'}
                                   )
                        ],
                        class_name='ms-0')
                    ])
                ],
                className='mb-1'),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Name',
                                  width='auto',
                                  class_name='text-info fw-bold me-0 d-inline',
                                  style={'font-size': '15px'})
                        ],
                        width='auto',
                        class_name='me-1'),
                    dbc.Col([
                        dbc.Select(id='dp-select-var-name-2',
                                   options=[],
                                   required=True,
                                   class_name='p-1 text-info d-inline',
                                   style={'font-size': '14px'}
                                   )
                        ],
                        class_name='ms-0')
                    ]),
                ])
            ],
            width=3,
            class_name='ms-1 me-2',
        ),
        
        dbc.Col(
            html.H4(html.I(className='bi bi-arrow-right-circle me-2 text-info')), # bi bi-caret-right-fill
            width='auto'
            ),
        
        dbc.Col([
            dbc.Label('Select Variable 2: ', width='auto', 
                      class_name='text-info me-0 fw-bold d-inline')
            ],
            width='auto'),
        dbc.Col([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Type', width='auto', 
                                  class_name='text-info fw-bold d-inline',
                                  style={'font-size': '15px'})
                        ],
                        width='auto',
                        class_name='me-1'),
                    dbc.Col([
                        dbc.Select(id='dp-select-var-type-3',
                                   options=[
                                       {'label': val, 'value': val}
                                       for val in var_types 
                                       if val not in ['time', 'geospatial']
                                       ],
                                   value='continuous',
                                   class_name='p-1 text-info d-inline',
                                   style={'font-size': '16px'}
                                   )
                        ],
                        class_name='ms-0')
                    ])
                ],
                className='mb-1'),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Name',
                                  width='auto',
                                  class_name='text-info fw-bold me-0 d-inline',
                                  style={'font-size': '15px'})
                        ],
                        width='auto',
                        class_name='me-1'),
                    dbc.Col([
                        dbc.Select(id='dp-select-var-name-3',
                                   options=[],
                                   required=True,
                                   class_name='p-1 text-info d-inline',
                                   style={'font-size': '14px'}
                                   )
                        ],
                        class_name='ms-0')
                    ]),
                ])
            ],
            width=3,
            class_name='ms-1 me-2',
        ),
        
        dbc.Col(
            html.H4(html.I(className=' bi bi-arrow-right-circle me-2 text-info')), # bi bi-caret-right-fill
            width='auto'
            ),
        dbc.Col(
            dbc.Button(['Show Relation'], 
                       id='button-show-relation',
                       active=True,
                       class_name='fw-bold text-white py-2 px-3 mx-1',
                       n_clicks=0,
                       style={'backgroundColor':'#FF9900'}), 
            width='auto'
            )
        ],
        class_name='g-2 mt-1 mb-1 align-items-center',
    )]
)

card_2var_graph_relation = dbc.Card([
    dbc.CardHeader([
        dbc.Row([
            dbc.Label('Sale_Price vs. Gr_Liv_Area', 
                      id='header-2var-names',
                      class_name='d-inline text-info fw-bold py-0')
            ])
        ],
        class_name='py-1'),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='fig-2var-relation', figure={}, 
                          className='p-0 h-100')
                ],
                # style={'height':'19rem'},
                class_name='p-0 overflow-auto',
                width=7),
            dbc.Col([
                html.Div([
                    dcc.Markdown([],
                        id='hp-test-result-interp',
                        className='text-info p-1 h-100 bg-transparent'
                        )
                    ]),
                html.Div([
                    dcc.Markdown([],
                        id= 'summary-stats-title',
                        className='text-info p-1 h-100 bg-transparent'
                        )
                    ]),
                html.Div([], id='summary-table'),
                html.Div([
                    dcc.Markdown([],
                        id= 'test-result-title',
                        className='text-info p-1 h-100 bg-transparent'
                        )
                    ]),
                html.Div([], id='result-table')
                ],
                # style={'height':'18rem'},
                class_name='h-100 px-0',
                width=5)
            ],
            justify='center',
            class_name='h-100 p-0')
        ],
        class_name='bg-transparent overflow-auto')
    ],
    class_name='mt-1 mb-1 p-1',
    style={'height':'30rem'})

# feature importance analysis header
card_3rd_part_header = dbc.Card([
    dbc.CardBody([
        dbc.Row([
            # image
            dbc.Col([
                html.Div([
                    html.Img(src='assets/Picture2.png',
                             height='130px',
                             className='ps-0 ms-0')]),
                ],
                class_name='ps-0 pe-1 ms-0',
                width=7),
            dbc.Col([
                html.Div([
                    html.H4('Feature Importance Analysis',
                            className='fw-bold', 
                            style={'color': '#485785'}
                            ),
                    dcc.Markdown(['''Finally, based on the above inference, which features do you think
                                    are most important to house price prediction? To answer this question, 
                                    this part provides Feature Importance Analysis results from the data 
                                    science project, where a Machine Learning Model was trained to predict house 
                                    price and the model coefficient data was extracted for feature ranking.'''],
                                  className='m-0 text-djustify',
                                  style={'line-height': '1.1'})
                    ],
                    className='ms-0')
                ],
                class_name='ps-0 ms-0',
                width=5)
            ])
        ], 
         class_name='m-0 pb-0 pt-2 ps-0 pe-2')
    ],
     class_name='border-body border-2 border-bottom-0 border-start-0 border-end-0 pb-0 px-2 mt-2 bg-gradient shadow-lg')

card_feat_impt_graph_desc = dbc.Card([
    dbc.CardHeader([
        dbc.Form(
            dbc.Row([
                dbc.Label('Select Impact Type', width='auto', 
                          class_name='text-info fw-bold my-0 py-0'),
                dbc.Col([
                    dbc.Select(id='dp-select-impact-type',
                               options=[
                                   {'label': val, 'value': val}
                                   for val in ['Combined', 'Positive', 'Negative']
                                   ],
                               value='Combined',
                               class_name='p-1 my-0 text-info',
                               style={'font-size': '15px'}
                               )],
                    width=2,
                    class_name='me-1 my-0 py-0',
                ),
                dbc.Label('Select Number of Features', width='auto',
                          class_name='text-info fw-bold my-0 py-0'),
                dbc.Col([
                    dbc.Select(id='dp-select-feat-num',
                               options=[{'label': val, 'value': val} for val in 
                                        ['5', '10', '15', '20', '25', '30', '35', '40', '45', '50']],
                               value='30',
                               required=True,
                               class_name='p-1 my-0 text-info',
                               style={'font-size': '15px'}
                               )],
                    width=2,
                    class_name='me-2 my-0 py-0',
                ),
                make_popover_feat_impt(
                    target='dp-select-impact-type', 
                    header='**Select Impact Type**', 
                    body_text='''- **Combined:** use ABSOLUTE coefficient value to rank features, higher 
                                value corresponds to higher ranking\n- **Positive:** only account for
                                features with POSITIVE coefficients, higher value corresponds to higher
                                ranking\n- **Negative:** only account for features with NEGATIVE coefficients, 
                                lower value corresponds to higher ranking'''),
                make_popover_feat_impt(
                    target='dp-select-feat-num', 
                    header='**Select Number of Features**', 
                    body_text='''- select the number of features with top impact ranking to visualize'''),                                  
                dbc.Col(
                    html.H4(html.I(className='bi bi-arrow-right-circle me-2 text-info')), # bi bi-caret-right-fill
                    width='auto',
                    class_name='my-0 pt-2 pb-0 mb-0'
                    ),
                dbc.Col([
                    dbc.Button(['Visualize'], 
                               id='button-visualize-feat-impt',
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
        class_name='py-1'),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='fig-feat-impt', 
                          # initialize figure
                          figure={}, 
                          className='p-0 h-100')
                ],
                # style={'height':'16rem'}, # adjust according to different variables
                class_name='p-0 overflow-auto',
                width=7),
            dbc.Col([
                html.H6([
                    'Model Building Pipeline for Feature Importance Analysis'
                    ],
                    className='fw-bold text-info ms-2 mb-0 pb-0'),
                html.Div([
                    html.Small(['part of '], className='d-inline'),
                    html.A(['House Price Prediction - Comprehensive Data Science Project'], 
                           href='https://github.com/ZNL0504/House-Price-Prediction',
                           className='small d-inline'),
                    ],
                    className='mt-0 mt-1 mb-2 py-0 text-end'),
                html.Div([
                    html.Img(src='assets/Picture3_1.png',
                             height='320px',
                             className='ps-0 ms-0')]),
                dbc.Card([
                    dbc.CardImg(src='assets/bg1.jpg', 
                                top=True,
                                class_name='pe-0',
                                style={'opacity': 0.3}),
                    dbc.CardImgOverlay([
                        dcc.Markdown([
                            '''###### **Keynotes about Model Building**\n- Only 80% of Kaggle TRAINING dataset was used
                             for model training to avoid data leakage in house sale price 
                             prediction.\n- Feature naming of Kaggle dataset is slightly 
                             different from AmesHousing dataset.\n- Multi-level nominal features
                             are ONE-HOT encoded before model training, each level is treated 
                             separately in feature ranking.'''
                            ],
                            className='text-info text-djustify pe-2')
                        ])
                    ],
                    class_name='mt-3 ms-2',
                    style={'width': '30rem'}
                    )
                ],
                # style={'height':'15rem'},
                class_name='h-100 ps-2',
                width=5)
            ],
            justify='center',
            class_name='h-100 p-0')
        ],
        class_name='bg-transparent overflow-auto')
    ],
    class_name='mt-0 mb-1 p-1',
    style={'height':'45rem'})
# =============================================================================
# Layout
# =============================================================================
layout = dbc.Container([
    # header
    dbc.Row([
        dbc.Col([
            html.H3('Dataset Exploration and Statistical Inference', 
                    className='fw-bold my-0', 
                    style={'display':'inline-block'})
            ],
            class_name='fw-bold text-center mt-3 mb-1 border-2 border-bottom shadow-sm',
            width={'size': 8, 'offset': 1}),
        dbc.Col([
            dbc.Button(['About this Page'],
                       id='button-about-pg-3',
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
                    '''This page looks into the whole dataset. On the one hand, explicit and convenient feature meaning reference 
                      can help understand the data intuitively. On the other hand, relationship among variables is interpreted analytically 
                      to provide a sense of how different variables are connected with each other. Specifically, this page contains three parts:
                      \n- in **Dataset Overview**, data documentation is introduced and combined with interactive dash data table (as table header tooltips) 
                      and plotly graphs, to give a whole picture of the dataset.\n- **Statistical Inference** depicts variable relationship 
                      with appropriate graph types and interprets hypothesis test to help figure out connections among variables.\n- **Fearure Importance Analysis**
                      introduces the results of a house sale price prediction machine learning model and tries to help highlight most important features in house price
                      prediction, which is a great chance to compare our understanding with what data really tells us.
                      \nTry to go through these parts step by step and check all the analytical results with a simple click.
                      '''
                    ])
                ]),
            dbc.ModalFooter(
                html.P('Part 2: Dataset Exploration and Statistical Inference'))
            ],
            id='modal-about-pg-3',
            size='lg',
            scrollable=True,
            is_open=False)
        ], 
        justify='center'),
    # first part header
    dbc.Row([
        card_first_part_header
        ]),
    # data table
    dbc.Row([
        dbc.Col([
            card_datatable_sidebar
            ],
            width=2),
        dbc.Col([
            card_datatable
            ],
            width=10)
        ],
        justify='center'),
    dbc.Row([
        form_feat_select
        ],
        justify='center'),
    dbc.Row([
        card_var_graph_desc
        ],
        justify='center'),
    # 2nd part header: statistical inference
    dbc.Row([
        card_2nd_part_header
        ]),
    dbc.Row([
        form_2feat_select
        ]),
    dbc.Row([
        card_2var_graph_relation
        ],
        justify='center'),
    # header of 3rd part: feature importance analysis
    dbc.Row([
        card_3rd_part_header
        ]),
    dbc.Row([
        card_feat_impt_graph_desc
        ],
        class_name='pb-5',
        justify='center')
    ],
    fluid=True
    )

# =============================================================================
# Callbacks
# =============================================================================
# about this page modal
@callback(
    Output('modal-about-pg-3', 'is_open'),
    [Input('button-about-pg-3', 'n_clicks')],
    [State('modal-about-pg-3', 'is_open')]
)
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

# highlight data in datatable
@callback(
    Output('table', 'style_header_conditional'),
    Output('table', 'style_data_conditional'),
    Input('cl-num-var', 'value'),
    Input('cl-cat-var', 'value'),
    Input('cl-other-var', 'value')
    )
def highlight_data(num_types, cat_types, other_types):
    style_header_cond = []
    style_data_cond = (
        [{
     'if': {'row_index': 'odd'},
     'backgroundColor': 'rgb(250, 250, 250)'
     }]
        +
        [
        {
         'if': {'column_id': col},
         'textDecoration': 'underline',
         'textDecorationStyle': 'dotted'
            } for col in feats_tooltip
        ]
        +
     data_bars(ames, 'Sale_Price')
     )
    if num_types + cat_types + other_types:
        for typ in num_types + cat_types + other_types:
            style_header_cond += [
                {
                 'if': {'column_id': col},
                 'backgroundColor': dtype_color_map[typ],
                 'color':'white'
                 } for col in var_type_dict[typ]
                ]
            if typ == 'continuous':
                style_data_cond += (
                    [{
                     'if': {'column_id': col},
                     'backgroundColor': dtype_color_map_rgba['continuous'],
                     } for col in var_type_dict['continuous']]
                    + 
                    [{
                     'if': {'column_id': 'Sale_Price'},
                     'backgroundColor': 'rgb(204,235,197)',
                     }]
                    )
            else:
                style_data_cond += [
                    {
                     'if': {'column_id': col},
                     'backgroundColor': dtype_color_map_rgba[typ],
                     } for col in var_type_dict[typ]
                    ]
    
    return style_header_cond, style_data_cond

# variable name dropdown options controled by variable type dropdown
@callback(
    Output('dp-select-var-name-1', 'options'),
    Input('dp-select-var-type-1', 'value'),
    )
def toggle_var_name_options_1(var_typ_1):
    options = [{'label': var, 'value': var} for var in var_type_dict[var_typ_1]]
    return options

@callback(
    Output('dp-select-var-name-2', 'options'),
    Input('dp-select-var-type-2', 'value'),
    )
def toggle_var_name_options_2(var_typ_2):
    options = [{'label': var, 'value': var} for var in var_type_dict[var_typ_2]]
    return options

@callback(
    Output('dp-select-var-name-3', 'options'),
    Input('dp-select-var-type-3', 'value')
    )
def toggle_var_name_options_3(var_typ_3):
    options = [{'label': var, 'value': var} for var in var_type_dict[var_typ_3]]
    return options

# variable name dropdown default value controled by options
@callback(
    Output('dp-select-var-name-1', 'value'),
    Input('dp-select-var-name-1', 'options'),
    )
def initialize_var_name_value_1(options_1):
    value = ('Sale_Price' if options_1[0]['value'] in var_type_dict['continuous'] 
                  else options_1[0]['value'])   
    return value

@callback(
    Output('dp-select-var-name-2', 'value'),
    Input('dp-select-var-name-2', 'options'),
    )
def initialize_var_name_value_2(options_2):
    value = options_2[0]['value']    
    return value

@callback(
    Output('dp-select-var-name-3', 'value'),
    Input('dp-select-var-name-3', 'options')
    )
def initialize_var_name_value_3(options_3):
    value = ('Sale_Price' if options_3[0]['value'] in var_type_dict['continuous'] 
                  else options_3[0]['value'])     
    return value

# card header, graph, data description: visualize single variable
@callback(
    Output('header-var-name', 'children'),
    Output('header-var-type', 'children'),
    Output('header-var-type', 'color'),
    Output('fig-single-var', 'figure'),
    Output('var-desc', 'children'),
    Input('dp-select-var-type-1', 'value'),
    Input('dp-select-var-name-1', 'value'),
    Input('button-visualize-data', 'n_clicks')
    )
def update_card_single_var(var_type, var_name, n):
    basic_h = 260 # basic height of figure
    if n == 0:
        fig = (px.histogram(ames, x='Sale_Price', nbins=50)
               .update_layout(
                   margin=dict(l=0, r=10, t=0, b=0),
                   height=basic_h
                   ))
        var_desc = ('**{}:**\n- {}\n- {}'
                    .format('Sale_Price',
                            var_desc_dict_full['Sale_Price'][0],
                            var_desc_dict_full['Sale_Price'][1]))
        return 'Sale_Price', 'continuous', dtype_color_map['continuous'], fig, var_desc
    elif 'button-visualize-data' == ctx.triggered_id:
        nbins = 50 if var_type == 'continuous' else None
        cat_order = ({var_name : list(ames[var_name].cat.categories)} 
                      if var_type == 'ordinal' else None)
        bar_gap = 0 if var_type == 'continuous' else 0.3
        fig_h = 500 if var_name in ['MS_SubClass', 'Neighborhood'] else basic_h
        add_text = True if var_type != 'continuous' else False
        
        fig = px.histogram(ames, x=var_name, nbins=nbins, 
                           text_auto=add_text,
                           category_orders=cat_order)
        fig.update_layout(
            margin=dict(l=0, r=10, t=0, b=0),
            bargap=bar_gap,
            height=fig_h
            )
        
        var_desc = ('**{}:**\n- {}\n- {}'
                    .format(var_name,
                            var_desc_dict_full[var_name][0],
                            var_desc_dict_full[var_name][1]))
    
        return var_name, var_type, dtype_color_map[var_type], fig, var_desc
    
    else:
        return dash.no_update

# card header, graph, hypothesis test result, interpretation: 
    # show relation between two variables
@callback(
    Output('header-2var-names', 'children'),
    # Output('header-var-type', 'children'),
    # Output('header-var-type', 'color'),
    Output('fig-2var-relation', 'figure'),
    Output('hp-test-result-interp', 'children'),
    Output('summary-stats-title', 'children'),
    Output('summary-table', 'children'),
    Output('test-result-title', 'children'),
    Output('result-table', 'children'),
    Input('dp-select-var-type-2', 'value'),
    Input('dp-select-var-name-2', 'value'),
    Input('dp-select-var-type-3', 'value'),
    Input('dp-select-var-name-3', 'value'),
    Input('button-show-relation', 'n_clicks')
    )
def update_card_show_relation(var_type_1, var_name_1, var_type_2, var_name_2, n):
    basic_h = 300 # basic height of figure
    if n == 0: # initialize the figure
        fig = (px.scatter(ames, x='Lot_Frontage', y='Sale_Price',
                          opacity=0.9,
                          marginal_x='histogram', marginal_y='histogram',
                          trendline='ols')
               .update_traces(
                   marker=dict(
                       line={'width': 0.3, 'color': '#0074D9'}
                       ))
               .update_layout(
                   margin=dict(l=0, r=10, t=0, b=0),
                   height=basic_h
                   ))
        header = 'Lot_Frontage vs. Sale_Price'
        relation_desc = stats_inference(df=ames,
                                        twolev_cat_vars_lst=twolev_cat_vars, 
                                        var_type_1='continuous', 
                                        var_type_2='continuous', 
                                        var_name_1='Lot_Frontage', 
                                        var_name_2='Sale_Price')[0]
        # relation_desc = ('**{}:**\n- {}\n- {}\n- {}'
        #                  .format('Statistical Inference', 
        #                          'test type',
        #                          'result',
        #                          'interpretation'))
        return header, fig, relation_desc, '', '', '', ''
        
    elif 'button-show-relation' == ctx.triggered_id:
        ### use function from self-defined module to make statistical inference
        relation_desc, summary_tbl, result_tbl = stats_inference(df=ames,
                                                                 twolev_cat_vars_lst=twolev_cat_vars, 
                                                                 var_type_1=var_type_1, 
                                                                 var_type_2=var_type_2, 
                                                                 var_name_1=var_name_1, 
                                                                 var_name_2=var_name_2)
        summary_stats_title = '**summary statistics:**' if summary_tbl != '' else ''
        test_result_title = '**hypothesis test result:**' if result_tbl != '' else ''
        
        ### update graph under variace conditions
        # numerical vs. numerical: scatter plot (exclude discrete vs discrete)
        # if both variables are discrete, plot heatmap, not scatter plot
        if set([var_type_1, var_type_2]) == set(['continuous', 'discrete']) or set([var_type_1, var_type_2]) == set(['continuous']):
            fig = (px.scatter(ames, x=var_name_1, y=var_name_2,
                              opacity=0.9,
                              marginal_x='histogram', marginal_y='histogram',
                              trendline='ols')
                   .update_traces(
                       marker=dict(
                           line={'width': 0.3, 'color': '#0074D9'}
                           ))
                   .update_layout(
                       margin=dict(l=0, r=10, t=0, b=0),
                       height=basic_h
                       ))
        # categorical + discrete vs. categorical + discrete: crosstab heatmap
        elif (var_type_1 in ['discrete', 'ordinal', 'nominal']) and (var_type_2 in ['discrete', 'ordinal', 'nominal']):
            crosstab = pd.crosstab(ames[var_name_1], ames[var_name_2])
            fig_h = (500 if (var_name_1 in ['MS_SubClass', 'Neighborhood']) 
                     or (var_name_2 in ['MS_SubClass', 'Neighborhood']) else basic_h)
            # color_scale = 'Teal_r'
            # color_mid = None
            # # change color_scale, color mid-point if both variables are 2-level
            # if (var_name_1 in twolev_cat_vars) and (var_name_2 in twolev_cat_vars):
            #     color_scale = 'balance_r'
            #     color_mid = 10
            fig = (px.imshow(crosstab, text_auto=True, aspect='auto',
                            color_continuous_scale='Teal_r',
                            # zmin=crosstab.min().min(),
                            # zmax=crosstab.max().max(),
                            # color_continuous_midpoint=color_mid,
                            labels=dict(color='cell count'))
                   .update_layout(
                       coloraxis=dict(
                           # cmax=crosstab.max().max(),
                           # cmid=10,
                           # cmin=crosstab.min().min(),
                           colorbar={'thickness': 12, 'x': 1.03}),
                       margin=dict(l=0, r=10, t=0, b=0),
                       height=fig_h
                       ))
        # # nominal vs. discrete: crosstab heatmap
        # elif set([var_type_1, var_type_2]) == set(['nominal', 'discrete']):
        #     crosstab = pd.crosstab(ames[var_name_1], ames[var_name_2])
        #     fig_h = (500 if (var_name_1 in ['MS_SubClass', 'Neighborhood']) 
        #              or (var_name_2 in ['MS_SubClass', 'Neighborhood']) else basic_h)
        #     fig = (px.imshow(crosstab, text_auto=True, aspect='auto',
        #                     color_continuous_scale='Teal_r',
        #                     labels=dict(color='cell count'))
        #            .update_layout(
        #                coloraxis=dict(colorbar={'thickness': 12, 'x': 1.03}),
        #                margin=dict(l=0, r=10, t=0, b=0),
        #                height=fig_h
        #                ))
        
        # continuous vs. categorical: box plot
        else: # one continuous var, one categorical var (exclude nominal vs. discrete)
            x_var = var_name_1 if var_type_1 in ['ordinal', 'nominal'] else var_name_2
            y_var = var_name_1 if var_type_1 == 'continuous' else var_name_2
            fig_h = 500 if x_var in ['MS_SubClass', 'Neighborhood'] else basic_h
            # if an ordinal var does not have values in each level, there will be
            # an error when creating box plot, just turn these vars back to 'object' dtype
            if x_var in ['Overall_Cond', 'Utilities', 'Exter_Qual', 'Pool_QC']:
                df = ames.copy()
                df[['Overall_Cond', 'Utilities', 'Exter_Qual', 'Pool_QC']] = df[['Overall_Cond', 'Utilities', 'Exter_Qual', 'Pool_QC']].astype('object')
            else:
                df = ames
            cat_orders = {x_var: list(ames[x_var].cat.categories)} if x_var in var_type_dict['ordinal'] else {}
            fig = (px.box(df, x=x_var, y=y_var, points='all',
                              color=x_var, category_orders=cat_orders,
                              color_discrete_sequence=px.colors.qualitative.G10)
                    .update_traces(
                        # jitter=0.2,
                        pointpos=0,
                        marker=dict(
                            opacity=0.5,
                            line={'width':0.2, 'color': 'white'},
                                    )
                        )
                    .update_layout(
                        showlegend=False,
                        margin=dict(l=0, r=10, t=0, b=0),
                        height=fig_h
                        ))
            
            # fig = go.Figure()
            # (fig.add_trace(
            #     go.Violin(x=ames[x_var], y=ames[y_var],
            #               points=False, opacity=0.7, # 'all'
            #               scalemode='width', showlegend=False,)
            #     )
            #     .update_traces(
            #         meanline_visible=True
            #         )
            #     .update_layout(
            #         margin=dict(l=0, r=10, t=0, b=0),
            #         height=basic_h
            #                 ))
            
        header = '{} vs. {}'.format(var_name_1, var_name_2)
        
        return header, fig, relation_desc, summary_stats_title, summary_tbl, test_result_title, result_tbl
    
    else:
        return dash.no_update

# graph: visualize feature importance
@callback(
    Output('fig-feat-impt', 'figure'),
    Input('dp-select-impact-type', 'value'),
    Input('dp-select-feat-num', 'value'),
    Input('button-visualize-feat-impt', 'n_clicks')
    )
def update_card_feat_impt(impact_type, num, n):
    if n == 0:
        coef_top_df = (feat_coef_df_new
                       .sort_values(by='coef', key = abs, ascending = False)[: 30]
                       .sort_values(by='coef').round(4)
                       .reset_index())
        fig = (px.bar(coef_top_df, x='coef', y='feature_v2',
                      color='coef',
                      color_continuous_midpoint=0,
                      color_continuous_scale='balance_r',
                      hover_name='feature_v2',
                      hover_data=['coef', 'original_feat_v2', 'feat_type'],
                      labels={'feature_v2': 'Feature', 'coef':'coefficient',
                              'original_feat_v2': 'original feature',
                              'feat_type': 'feature type'},
                      title='<b>top 30 features</b> | Lasso Model (alpha = 0.0002)',
                      text_auto=True)
               .update_traces(textposition='outside')
               .update_layout(
                   coloraxis_showscale=False,
                   yaxis=dict(
                       showspikes=True,
                       showgrid=False),
                   xaxis=dict(title={'text': 'Coefficient'}, dtick=1, showgrid=False),                        
                   paper_bgcolor='rgba(0,0,0,0)',
                   plot_bgcolor='rgba(0,0,0,0)',
                   margin=dict(l=0, r=0, t=30, b=0),
                    height=590
                   ))
        return fig
    elif 'button-visualize-feat-impt' == ctx.triggered_id:
        sort_key = abs if impact_type == 'Combined' else None
        ascend = True if impact_type == 'Negative' else False
        if num in ['5', '10', '15']:
            fig_h = 350
        else:
            fig_h = 350 + 80 * (int(num) - 15)/5
        
        if impact_type == 'Combined':
            title= '<b>top {} features</b> | Lasso Model (alpha = 0.0002)'.format(num)
        else:
            impact = 'positively' if impact_type == 'Positive' else 'negatively'
            title = '<b>top {} features that {} impact house price</b> | Lasso Model (alpha = 0.0002)'.format(num, impact)
        coef_top_df = (feat_coef_df_new
                       .sort_values(by='coef', key = sort_key, ascending = ascend)[: int(num)]
                       .sort_values(by='coef').round(4)
                       .reset_index())
        fig = (px.bar(coef_top_df, x='coef', y='feature_v2',
                      color='coef',
                      color_continuous_midpoint=0,
                      color_continuous_scale='balance_r',
                      hover_name='feature_v2',
                      hover_data=['coef', 'original_feat_v2', 'feat_type'],
                      labels={'feature_v2': 'Feature', 'coef':'coefficient',
                              'original_feat_v2': 'original feature',
                              'feat_type': 'feature type'},
                      title=title,
                      text_auto=True)
               .update_traces(textposition='outside')
               .update_layout(
                   coloraxis_showscale=False,
                   yaxis=dict(
                       showspikes=True,
                       showgrid=False),
                   xaxis=dict(title={'text': 'Coefficient'}, dtick=1, showgrid=False),                        
                   paper_bgcolor='rgba(0,0,0,0)',
                   plot_bgcolor='rgba(0,0,0,0)',
                   margin=dict(l=0, r=0, t=30, b=0),
                   height=fig_h
                   ))
        return fig
    
    else:
        return dash.no_update









