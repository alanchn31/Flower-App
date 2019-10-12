import os
from flask import Flask, request, render_template, flash, \
    redirect, url_for

from utils import *

import torch

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index')
def index():
    return render_template('base.html', title='Flower App')

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, \
                                         post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
       if 'file' not in request.files:
           flash('No file part')
           return redirect(request.url)
       file = request.files['file']
       if file.filename == '':
           flash('No selected file')
           return redirect(request.url)
       if file and allowed_file(file.filename):
           file_path = file.filename
           file.save(file_path)
           return redirect(url_for('uploaded_v2', file_path=file_path))

@app.route('/uploaded/<file_path>')
def uploaded(file_path):
    #file_path = request.args['file_path']
    model_path = 'model/udacity_proj_densenet_121_model.pt'
    model = torch.load(model_path,map_location='cpu')
    plot_bar(file_path, model)
    return redirect(url_for('result'))

@app.route('/uploaded_v2/<file_path>')
def uploaded_v2(file_path):
    #file_path = request.args['file_path']
    model_path = 'model/udacity_proj_densenet_121_model.pt'
    model = torch.load(model_path,map_location='cpu')
    script, div = plot_bar_v2(file_path, model)
    return render_template('plot_v2.html', title='Flower App', the_div=div, the_script=script)

@app.route('/result')
def result():
    return render_template('plot.html', title='Flower App')

if __name__ == "__main__":
    app.run(debug=True)