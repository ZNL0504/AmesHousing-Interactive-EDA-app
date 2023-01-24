# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 19:15:01 2022

@author: ZNL
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import stats_inference_func

app = dash.Dash(__name__, use_pages=True, 
                external_stylesheets=[dbc.themes.MORPH, dbc.icons.BOOTSTRAP])
server = app.server

# =============================================================================
# Dashboard Header layout
# =============================================================================
pages = list(dash.page_registry.values())
print(pages)
page_bar = dbc.Nav(
            [
                dbc.NavLink(
                    [
                        html.Div(pages[0]['name'], className='my-0 p-0'),
                    ],
                    href=pages[0]['path'],
                    active='exact',
                    class_name='me-5 my-0'
                )
            ] 
            + 
            [
                dbc.NavLink(
                    [
                        html.Div(page['name'], className='my-0 p-0'),
                    ],
                    href=page['path'],
                    active='exact',
                    class_name='my-0'
                )
                for page in pages[1:]
            ],
            horizontal=True,
            pills=True,
            class_name='p-2 my-0 bg-light bg-gradient',
)

header = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [dbc.Col(
                    [
                        html.Div(
                            [
                                html.H2('House Sale Evaluation', 
                                        className='fw-bolder text-white my-0'),
                                html.P('Interactive Data Visualization & Analysis with AmesHousing Dataset',
                                       className='fw-light text-light my-0')
                            ],
                            id='app-title'
                            )
                    ],
                    md=True,
                    align='center'
                    )], 
                align='left'
                ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            page_bar
                        ])
                ], 
                align='center')
        ], fluid=True
    ),
    color='primary',
    class_name='bg-gradient',
    sticky='top'
   )

# =============================================================================
# Dashboard overall layout
# =============================================================================
app.layout = dbc.Container([
    dbc.Row([
        header
        ]),
    dbc.Row([
        dbc.Col(
            [
                dash.page_container
            ])
        ])
   ], fluid=True)

# app.layout = dbc.Container([
#     dbc.Row([
#         dbc.Col(html.Div('Python Multipage App with Dash',
#                          style={'fontSize':50, 'textAlign':'center'}))
#     ]),

#     html.Hr(),

#     dbc.Row(
#         [
#             dbc.Col(
#                 [
#                     navbar
#                 ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2),

#             dbc.Col(
#                 [
#                     dash.page_container
#                 ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
#         ]
#     )
# ], fluid=True)


if __name__ == "__main__":
    app.run(debug=True)
