import IPython
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64


def run(img_str):
    #decode to image
    decimg = base64.b64decode(img_str.split(',')[1], validate=True)
    decimg = Image.open(BytesIO(decimg))
    decimg = np.array(decimg, dtype=np.uint8);
    decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)

    #############your process###############

    out_img = cv2.Canny(decimg,100,200)
    #out_img = decimg

    #############your process###############

    #encode to string
    _, encimg = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    img_str = encimg.tostring()
    img_str = "data:image/jpeg;base64," + base64.b64encode(img_str).decode('utf-8')
    return IPython.display.JSON({'img_str': img_str})
