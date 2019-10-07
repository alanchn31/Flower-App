import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import column
from bokeh.palettes import Spectral7
from bokeh.plotting import figure
from bokeh.embed import components
from PIL import Image

with open('data/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Resize where shortest side is 256px, keeping aspect ratio
    minside = 256
    img = Image.open(image)
    imagex, imagey = img.size
    aspect = float(imagex) / float(imagey)

    if imagex <= imagey:
        width = 256
        height = int(width / aspect)
    else:
        height = 256
        width = int(height * aspect)

    img = img.resize((width, height), Image.ANTIALIAS)

    # Crop out center 224 x 224
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    img = img.crop((left, top, right, bottom))

    # Convert image to numpy array
    np_image = np.array(img)
    np_image = np_image / 255

    # Normalize image
    np_image -= [0.485, 0.456, 0.406]
    np_image /= [0.229, 0.224, 0.225]

    # Transpose array:
    result = np_image.transpose(-1, 0, 1)

    return result


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, topk=5, train_on_gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    model.eval()
    image = process_image(image_path)
    image = torch.from_numpy(image)
    if train_on_gpu:
        image = image.cuda()
    image = image.float().unsqueeze(0)
    out = model.forward(image)
    logps = F.log_softmax(out)
    ps = torch.exp(logps)
    probs, classes = ps.topk(5, dim=1)
    if train_on_gpu:
        probs = list(probs.squeeze(0).cpu().detach().numpy())
        classes = list(classes.squeeze(0).cpu().detach().numpy())
    else:
        probs = list(probs.squeeze(0).detach().numpy())
        classes = list(classes.squeeze(0).detach().numpy())
    idx_class_mapping = dict((v, k) for k, v in model.class_to_idx.items())
    classes = list(map(lambda x: idx_class_mapping[x], classes))
    return probs, classes

def plot_bar(image_path, model):
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(APP_ROOT,'static/')
    destination = "/".join([target,"plot_bar.png"])
    result = process_image(image_path)
    res = torch.from_numpy(result)
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1 = imshow(res, ax1)
    probs, classes = predict(image_path,model)
    ax2.barh(np.arange(len(probs)), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(probs)))
    classes = list(map(lambda x: cat_to_name[x], classes))
    ax2.set_yticklabels(classes, size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig(destination,bbox_inches='tight',pad_inches = 0.0)

def plot_bar_v2(image_path, model):
    result = process_image(image_path)
    res = torch.from_numpy(result)
    fig, ax = plt.subplots(figsize=(6,9))
    imshow(res, ax)
    plt.axis('off')
    destination = "/".join([target,"flower.png"])
    plt.savefig(destination, bbox_inches='tight',pad_inches = 0.0)
    probs, classes = predict(image_path,model)
    classes = list(map(lambda x: cat_to_name[x], classes))

    source = ColumnDataSource(data=dict(classes=classes, probs=probs, color=Spectral7))
    hover = HoverTool(tooltips=[
        ("index", "$index"),
        ("class", "@classes"),
        ("prob", "$y"),
    ])
    p = figure(x_range=classes, y_range=(0,1), plot_height=350, 
            title="Probabilities of Flower Classes Predictions",
            toolbar_location=None, tools=[hover])

    p.vbar(x='classes', top='probs', width=0.9, color='color', legend="classes", source=source)

    p.xgrid.grid_line_color = None
    p.legend.orientation = "vertical"
    p.legend.location = "top_right"
    script, div = components(p)
    return script, div
