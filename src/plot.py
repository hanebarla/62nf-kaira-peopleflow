from src.model import CANNet2s
from src.utils import fix_model_state_dict, NormalizeQuiver, tm_output_to_dense

import IPython
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import torch
import torch.nn.functional as F
import torchvision
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.colors import Normalize


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = CANNet2s(load_weights=True)
checkpoint = torch.load('weights/model_best.pth.tar')
model.load_state_dict(fix_model_state_dict(checkpoint['state_dict']))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
print(device)

input_img = {
    'first': torch.zeros((3, 360, 640)),
    'second': torch.zeros((3, 360, 640))
}

plotcm = cm.Reds
norm = Normalize()


def quiver2img(qv):
    fig = Figure(dpi=200)
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    colors = np.sqrt(qv[2]*qv[2] + qv[3]*qv[3])
    v_length = colors.copy()
    norm.autoscale(colors)
    ax.quiver(qv[0],
              qv[1],
              qv[2] / v_length,
              qv[3] / v_length,
              color=plotcm(norm(colors)),
              angles='xy', scale_units='xy', scale=1,
              headlength=1.2, headaxislength=1.08,
              pivot='mid')
    ax.set_ylim(0, 45)
    ax.set_xlim(0, 80)
    ax.set_aspect('equal')

    canvas.draw()
    im = np.array(fig.canvas.renderer.buffer_rgba())
    im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
    h, w, c = im.shape
    im = cv2.resize(im, (640, int(640*(h/w))))

    return im


def run(img_str):
    # decode to image
    decimg = base64.b64decode(img_str.split(',')[1], validate=True)
    decimg = Image.open(BytesIO(decimg))
    decimg = np.array(decimg, dtype=np.uint8)
    decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
    h, w, c = decimg.shape
    re_decimg = decimg[(int(h/2)-180):(int(h/2)+180), (int(w/2)-320):(int(w/2)+320)]

    input_img['second'] = transform(re_decimg)

    prev_img = input_img['first'].to(device).unsqueeze(0)
    img = input_img['second'].to(device).unsqueeze(0)

    with torch.no_grad():
        pred = model(prev_img, img)

    pred_num = pred[0, :, :, :].detach().cpu().numpy()
    pred_quiver = NormalizeQuiver(pred_num)
    quiver_img = quiver2img(pred_quiver)
    pred_num = pred_num.transpose((1, 2, 0))
    pred_dense = tm_output_to_dense(pred_num)

    pred_dense = np.array(pred_dense*(255/np.max(pred_dense)), np.uint8)
    pred_dense = cv2.resize(pred_dense, (640, 360), interpolation=cv2.INTER_CUBIC)
    cm_dense = cv2.applyColorMap(pred_dense, cv2.COLORMAP_JET)

    plot_img = np.concatenate([cm_dense, quiver_img], 0)

    input_img['first'] = input_img['second']

    # encode to string
    _, encimg = cv2.imencode(".jpg", plot_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    img_str = encimg.tostring()
    img_str = "data:image/jpeg;base64," + base64.b64encode(img_str).decode('utf-8')
    return IPython.display.JSON({'img_str': img_str})
