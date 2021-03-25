from collections import OrderedDict
from IPython.display import display, Javascript
from google.colab.output import eval_js

import numpy as np


def use_cam(quality=0.8):
    js = Javascript('''
    async function useCam(quality) {
      const div = document.createElement('div');
      document.body.appendChild(div);
      //video element
      const video = document.createElement('video');
      video.style.display = 'None';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      //canvas for display. frame rate is depending on display size and jpeg quality.
      display_size = 640
      const src_canvas = document.createElement('canvas');
      src_canvas.width  = display_size;
      src_canvas.height = display_size * video.videoHeight / video.videoWidth;
      const src_canvasCtx = src_canvas.getContext('2d');
      src_canvasCtx.translate(src_canvas.width, 0);
      src_canvasCtx.scale(-1, 1);
      div.appendChild(src_canvas);

      const dst_canvas = document.createElement('canvas');
      dst_canvas.width  = src_canvas.width;
      dst_canvas.height = src_canvas.height * 2;
      const dst_canvasCtx = dst_canvas.getContext('2d');
      div.appendChild(dst_canvas);

      //exit button
      const btn_div = document.createElement('div');
      document.body.appendChild(btn_div);
      const exit_btn = document.createElement('button');
      exit_btn.textContent = 'Exit';
      var exit_flg = true
      exit_btn.onclick = function() {exit_flg = false};
      btn_div.appendChild(exit_btn);

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      var send_num = 0
      // loop
      _canvasUpdate();
      async function _canvasUpdate() {
            src_canvasCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, src_canvas.width, src_canvas.height);
            if (send_num<1){
                send_num += 1
                const img = src_canvas.toDataURL('image/jpeg', quality);
                const result = google.colab.kernel.invokeFunction('notebook.run', [img], {});
                result.then(function(value) {
                    parse = JSON.parse(JSON.stringify(value))["data"]
                    parse = JSON.parse(JSON.stringify(parse))["application/json"]
                    parse = JSON.parse(JSON.stringify(parse))["img_str"]
                    var image = new Image()
                    image.src = parse;
                    image.onload = function(){dst_canvasCtx.drawImage(image, 0, 0)}
                    send_num -= 1
                })
            }
            if (exit_flg){
                requestAnimationFrame(_canvasUpdate);
            }else{
                stream.getVideoTracks()[0].stop();
            }
      };
    }
    ''')
    display(js)
    data = eval_js('useCam({})'.format(quality))


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


ChannelToLocation = ['aboveleft', 'above', 'aboveright',
                     'left', 'center', 'right',
                     'belowleft', 'below', 'belowright']


def NormalizeQuiver(output):
    output_num = output

    o_max = np.max(output_num)
    heats_u = np.zeros_like(output_num[0, :, :])
    heats_v = np.zeros_like(output_num[0, :, :])

    for i in range(9):
        out = output_num[i, :, :]
        # mean = np.mean(out)
        # std = np.std(out)
        # print("{} max: {}".format(ChannelToLocation[i], np.max(out)))
        # print("{} min: {}".format(ChannelToLocation[i], np.min(out)))
        heatmap = np.array(255*(out/o_max), dtype=np.uint8)

        if i == 0:
            heats_u -= heatmap/(255 * np.sqrt(2))
            heats_v += heatmap/(255 * np.sqrt(2))
        elif i == 1:
            heats_v += heatmap/255
        elif i == 2:
            heats_u += heatmap/(255 * np.sqrt(2))
            heats_v += heatmap/(255 * np.sqrt(2))
        elif i == 3:
            heats_u -= heatmap/255
        elif i == 5:
            heats_u += heatmap/255
        elif i == 6:
            heats_u -= heatmap/(255 * np.sqrt(2))
            heats_v -= heatmap/(255 * np.sqrt(2))
        elif i == 7:
            heats_v -= heatmap/255
        elif i == 8:
            heats_u += heatmap/(255 * np.sqrt(2))
            heats_v -= heatmap/(255 * np.sqrt(2))

    x, y = heats_u.shape[0], heats_u.shape[1]
    imX = np.zeros_like(heats_u)
    for i in range(y):
        imX[:, i] = np.linspace(x, 0, x)
    imY = np.zeros_like(heats_v)
    for i in range(x):
        imY[i, :] = np.linspace(0, y, y)

    v_leng = np.sqrt(heats_u * heats_u + heats_v * heats_v)
    v_leng_true = v_leng > 0
    imX = imX[v_leng_true]
    imY = imY[v_leng_true]
    # heats_u_cut = heats_u[v_leng_true] / v_leng[v_leng_true]
    # heats_v_cut = heats_v[v_leng_true] / v_leng[v_leng_true]
    heats_u_cut = heats_u[v_leng_true]
    heats_v_cut = heats_v[v_leng_true]
    # cut_lengs = np.sqrt(heats_u_cut * heats_u_cut + heats_v_cut * heats_v_cut)
    # heats_u_cut = heats_u_cut / cut_lengs
    # heats_v_cut = heats_v_cut / cut_lengs

    return (imY, imX, heats_u_cut, heats_v_cut)


def tm_output_to_dense(output):
    output_sum = np.zeros_like(output[:, :, 0])
    for i in range(9):
        output_sum += output[:, :, i]

    temp_max = np.max(output_sum)
    output_sum /= temp_max

    return output_sum
