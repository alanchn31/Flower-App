import os
from flask import Flask, request, render_template, flash, \
    redirect, url_for

from utils import *

import torch

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index')
def index():
    return render_template('base.html', title='Flower App')

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
           file.save(file.filename)
           return redirect(url_for('.uploaded', file_path=file_path))

@app.route('/uploaded')
def uploaded():
    file_path = request.args['file_path']
    model_path = 'model/udacity_proj_densenet_121_model.pt'
    model = torch.load(model_path,map_location='cpu')
    plot_bar(file_path, model)
    return redirect(url_for('result'))

@app.route('/result')
def result():
    return render_template('plot.html', title='Flower App')

if __name__ == "__main__":
    app.run(debug=True)