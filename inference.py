import os
import sys
sys.path.append('core')

from argparse import ArgumentParser
from collections import OrderedDict

import cv2
import numpy as np
import torch

from raft import RAFT
from utils import flow_viz
from utils import frame_utils
#from utils.utils import InputPadder

FLO_TAG = 202021.25

def tensor2img(img_t):
    img = img_t[0].detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))

    return img

def save_flo(flow, filename):
    flow = tensor2img(flow)
    with open(filename, 'wb') as f:
        tag = np.array([FLO_TAG], dtype=np.float32)

        (height, width) = flow.shape[0:2]
        w = np.array([width], dtype=np.int32)
        h = np.array([height], dtype=np.int32)
        tag.tofile(f)
        w.tofile(f)
        h.tofile(f)
        flow.tofile(f)

def frame_preprocess(frame, device):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    return frame


def vizualize_flow(img, flo, save, counter):
    # permute the channels and change device is necessary
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    flo = cv2.cvtColor(flo, cv2.COLOR_RGB2BGR)

    # concatenate, save and show images
    img_flo = np.concatenate([img, flo], axis=0)
    if save:
        cv2.imwrite(f"demo_frames/frame_{str(counter)}.jpg", img_flo)
    #cv2.imshow("Optical Flow", img_flo / 255.0)
    k = cv2.waitKey(25) & 0xFF
    if k == 27:
        return False
    return True


def get_cpu_model(model):
    new_model = OrderedDict()
    # get all layer's names from model
    for name in model:
        # create new name and update new model
        new_name = name[7:]
        new_model[new_name] = model[name]
    return new_model


def inference(args):
    # get the RAFT model
    model = RAFT(args)
    # load pretrained weights
    pretrained_weights = torch.load(args.model)

    save = args.save
    if save:
        if not os.path.exists("demo_frames"):
            os.mkdir("demo_frames")

    if torch.cuda.is_available():
        device = "cuda"
        # parallel between available GPUs
        model = torch.nn.DataParallel(model)
        # load the pretrained weights into model
        model.load_state_dict(pretrained_weights)
        model.to(device)
    else:
        device = "cpu"
        # change key names for CPU runtime
        pretrained_weights = get_cpu_model(pretrained_weights)
        # load the pretrained weights into model
        model.load_state_dict(pretrained_weights)

    # change model's mode to evaluation
    model.eval()

    video_path = args.video
    # capture the video and get the first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame_1 = cap.read()

    # frame preprocessing
    frame_1 = frame_preprocess(frame_1, device)

    counter = 0
    optical_flow_path = args.output_path
    viz = args.viz
    with torch.no_grad():
        while True:
            # read the next frame
            ret, frame_2 = cap.read()
            if not ret:
                break
            # preprocessing
            frame_2 = frame_preprocess(frame_2, device)
            #padder = InputPadder(frame_1.shape)
            #frame_1, frame_2 = padder.pad(frame_1[None].cuda(), frame_2[None].cuda())
            # predict the flow
            flow_low, flow_up = model(frame_1, frame_2, iters=args.iters, test_mode=True)
            #flow = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()
            output_file = os.path.join(optical_flow_path, 'frame%06d.flo' % (counter))

            if not os.path.exists(optical_flow_path):
                os.makedirs(optical_flow_path)

            #frame_utils.writeFlow(output_file, flow_up)
            save_flo(flow_up, output_file)

            if viz:
                # transpose the flow output and convert it into numpy array
                ret = vizualize_flow(frame_1, flow_up, save, counter)
                if not ret:
                    break
            frame_1 = frame_2
            counter += 1

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--video", type=str, default="./videos/car.mp4")
    parser.add_argument("--output_path", type=str, default="./optical_flow")
    parser.add_argument("--save", action="store_true", help="save demo frames")
    parser.add_argument("--viz", action="store_true", help="show visualization")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    args = parser.parse_args()
    inference(args)


if __name__ == "__main__":
    main()