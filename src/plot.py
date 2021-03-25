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
import matplotlib.pyplot as plt


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
    pred_num = pred_num.transpose((1, 2, 0))
    pred_dense = tm_output_to_dense(pred_num)

    pred_dense = np.array(pred_dense*(255/np.max(pred_dense)), np.uint8)
    pred_dense = cv2.resize(pred_dense, (640, 360), interpolation=cv2.INTER_CUBIC)

    input_img['first'] = input_img['second']

    # encode to string
    _, encimg = cv2.imencode(".jpg", pred_dense, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    img_str = encimg.tostring()
    img_str = "data:image/jpeg;base64," + base64.b64encode(img_str).decode('utf-8')
    return IPython.display.JSON({'img_str': img_str})
