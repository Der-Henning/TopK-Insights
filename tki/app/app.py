"""Module containing a dash application as interface for tki"""
from typing import NoReturn, List, Tuple, Any
import base64
import io
import pickle
import pandas as pd
import dash
from dash import Dash, html, dcc, Output, Input, dash_table, ALL
from dash.exceptions import PreventUpdate

from tki.insights import InsightResult
from tki.app.plots import Plot

class Upload(dcc.Upload):
    """Formatted Upload control"""
    def __init__(self, *args, **kwargs):
        super().__init__(
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select File')
            ]),
            style={
                'width': '90%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px auto'
            },
            *args, **kwargs)

class App():
    """Web application providing an interface for TKI

    Parameters
    ----------
    port : int
        Port used to serve the application. Defaults to 8050
    debug : bool
        Tun on debug mode. Defaults to False
    """
    # TODO:
    # Currently working: Upload and visualize Insight Results
    # Create Interface to initialize TKI and run the calculation
    # Global variables = bad. Only for single user sessions.
    # For multi session capability propably a database (Redis?) is needed.
    def __init__(self, port: int = 8050, debug: bool = False) -> None:
        self._port = port
        self._debug = debug

        # global variables
        self._data: pd.DataFrame = None
        self._insights: List[InsightResult] = None
        self._dim_setting = {}

        # Initialize Dash Application
        self._app = Dash("TKI")
        self._app.title = "Top-K Insights"

        # 2-Tab Layout
        self._app.layout = html.Div([
            html.H1("Top-K Insights"),
            dcc.Tabs([
                dcc.Tab(label='Analysis', children=[
                    html.Div([
                        html.H4('Upload Data'),
                        Upload(
                            id='upload-data',
                            multiple=False),
                        dash_table.DataTable(
                            id='data-table',
                            style_table={'overflowX': 'auto'},
                            column_selectable="multi",
                            page_current=0,
                            page_size=10),
                    ]),
                    html.Div(id='dimension-selector'),
                    html.Button('Calculate Insights', id='calc-insights')
                ]),
                dcc.Tab(label='Results', children=[
                    html.H4('Upload Insight Result file'),
                    Upload(
                        id='upload-results',
                        multiple=False),
                    html.Div(id='insight-container')
                ])
            ])
        ])

        # Callbacks
        # Upload data as .csv
        self._app.callback(
            [Output('data-table', 'data'),
            Output('data-table', 'columns')],
            Input('upload-data', 'contents')
        )(self._upload_data)

        # Manage selected dimensions
        self._app.callback(
            Output('dimension-selector', 'children'),
            Input('data-table', 'selected_columns')
        )(self._update_dimension_selector)

        # On dimension dropdown select save selection
        self._app.callback(
            Output('insight-container', 'style'),
            Input({'type': 'dimension-dropdown', 'index': ALL}, 'value')
        )(self._update_dimension_dropdown)

        # Upload Insight Results as .pkl
        self._app.callback(
            Output('insight-container', 'children'),
            Input('upload-results', 'contents')
        )(self._update_graphs)

    def _upload_data(self, content: str) -> Tuple[dict, dict]:
        if content is None:
            raise PreventUpdate
        _, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        self._data = pd.read_csv(io.BytesIO(decoded))
        return (self._data.to_dict('records'),
            [{"name": i, "id": i, "selectable": True} for i in self._data.columns])

    def _update_dimension_selector(self, columns: List[str]
        ) -> List[dash.development.base_component.Component]:
        if columns is None:
            raise PreventUpdate
        dims = [html.Div([
            html.Label(col),
            dcc.Dropdown(
                ['Nominal', 'Cardinal'],
                self._dim_setting[col] if col in self._dim_setting else None,
                id={'type': 'dimension-dropdown', 'index': col})
        ]) for col in columns]
        for key in list(self._dim_setting.keys()):
            if not key in columns:
                del self._dim_setting[key]
        return [html.H4('Edit Dimensions'), *dims]

    def _update_dimension_dropdown(self, _: List[Any]) -> None:
        for dropdown in dash.callback_context.inputs_list[0]:
            self._dim_setting[dropdown['id']['index']] = dropdown['value']
        raise PreventUpdate

    def _update_graphs(self, content: str) -> List[html.Div]:
        if content is None:
            raise PreventUpdate
        _, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        self._insights = pickle.load(io.BytesIO(decoded))
        return self._generate_insight_figures()

    def _generate_insight_figures(self) -> List[html.Div]:
        return [html.Div([
            html.Div([
                html.H4(f"{idx}) {type(insight.insight).__name__} - "f"Score: {insight.score:.2f}"),
                html.Div(f"{(insight.sibling_group, insight.composite_extractor)}"),
                dcc.Graph(figure=Plot(insight))]),
            html.Div([
                html.H4("Details:"),
                html.Div(insight.p_value)], style={'display': 'inline-block'})
        ]) for idx, insight in enumerate(self._insights, 1)]

    def run(self) -> NoReturn:
        """Starts webserver"""
        self._app.run(debug=self._debug, port=self._port)


if __name__ == "__main__":
    app = App()
    app.run()
