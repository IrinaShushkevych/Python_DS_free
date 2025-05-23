from dash import html, Dash, dcc, Input, Output, callback
import plotly.express as px
from datetime import date

app = Dash(__name__)

df = px.data.election()

fig = px.bar(df, x='district', y='total', color='winner', title='Election Results by District')

app.layout = html.Div(children=[
    html.H1(children='Election Results by District'),
    html.Div(children='An interactive bar chart showing election results in various districts.'),
    dcc.Graph(id='election-bar-chart', figure=fig),
    html.Div(id='output-container-date-picker'),
    dcc.DatePickerSingle(id='date_picker-single', date='2025-05-22',display_format='DD-MM-YYYY'),
    dcc.DatePickerRange(id='date_picker_range', start_date='2025-05-01', end_date='2025-05-31', display_format='DD-MM-YYYY')
    
    
])


@callback(
    Output('output-container-date-picker', 'children'),
    Input('date_picker_range', 'start_date'),
    Input('date_picker_range', 'end_date'))
def update_output(start_date, end_date):
    string_prefix = 'You have selected: '
    if start_date is not None:
        start_date_object = date.fromisoformat(start_date)
        start_date_string = start_date_object.strftime('%B %d, %Y')
        string_prefix = string_prefix + 'Start Date: ' + start_date_string + ' | '
    if end_date is not None:
        end_date_object = date.fromisoformat(end_date)
        end_date_string = end_date_object.strftime('%B %d, %Y')
        string_prefix = string_prefix + 'End Date: ' + end_date_string
    if len(string_prefix) == len('You have selected: '):
        return 'Select a date to see it displayed here'
    else:
        return string_prefix
    

if __name__ == '__main__':
    app.run(debug=True)