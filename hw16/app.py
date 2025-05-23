from dash import html, Dash, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
from PIL import Image, ImageOps
import base64
import io


app = Dash(__name__)

app.layout = html.Div(children=[ 
    html.H1(children='Класифікація зображень'),
    html.Div(children=[
        dcc.Upload(
            id='upload-img', 
            children=html.Button('Завантажити зображення'),
            style={
                'width': '300px', 
                'height': '60px', 
                'lineHeight': '60px', 
                'margin': '10px'
            },
            multiple=False,
            accept='image/jpeg, image/png, image/jpg'
        ),
        html.IMG(id='output-upload-img'),

    ])
])

def parse_contents(contents):
    if contents:
        _, content_string = contents.split(',')
        return base64.b64decode(content_string)

@app.callback(
    Output('output-upload-img', 'src'),
    Input('upload-img', 'contents'),
    State('upload-img', 'filename'),
    State('upload-img', 'last_modified')
)
def upload_image(contents, filename, date):
    if contents is None:
        raise PreventUpdate
    image_data = parse_contents(contents)
    if image_data is None:
        raise PreventUpdate
    image = Image.open(io.BytesIO(image_data))
    
    resized_image = ImageOps.grayscale(image.resize((28,28)))
    buffered = io.BytesIO()
    resized_image.save(buffered, format='PNG')

    encoded_image = base64.b64encode(buffered.getvalue())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())


if (__name__ == '__main__'):
    app.run(debug=True)