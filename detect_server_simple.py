# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import json
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import procbridge


class Detector():

    def __init__(self,
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            dnn=False,  # use OpenCV DNN for ONNX inference
            weights=ROOT / 'yolov5s.pt',  # model path or triton URL
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            half=False,  # use FP16 half-precision inference
            exist_ok=False,  # existing project/name ok, do not increment
    ):
        self.save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run

        # Load model
        device = select_device(device)
        self.weights = weights
        self.model = Detector.load_model(weights, device, dnn, data, half)

    @staticmethod
    @smart_inference_mode()
    def load_model(weights, device, dnn, data, fp16):
        return DetectMultiBackend(weights=weights, device=device, dnn=dnn, data=data, fp16=fp16, key='0123456789abcdef'.encode('utf-8'), iv=b'qqqqqqqqqqqqqqqq')

    @smart_inference_mode()
    def run(self,
            source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            vid_stride=1,  # video frame-rate stride
            weights=None,
            model=None,
    ):
        weights = weights if weights else self.weights
        model = model if model else self.model

        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download
        print(source)

        # Load model
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        tmp=(1 if pt or model.triton else bs, 3, *imgsz)
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        id2type = {0: 'person', 1: 'personcart'}
        res_list = list()
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape)
                for item in det:
                    if item[5]==0 or item[5]==1:
                        res = dict()
                        # pos = [f'{float(i):.02f}' for i in item[:4]]
                        pos = [float((i*100).round())/100 for i in item[:4]]
                        res['position'] = pos
                        res['type'] = id2type[int(item[5])]
                        res['confidence'] = float((item[4]*100).round())/100
                        LOGGER.info(res)
                        res_list.append(res)
                        # LOGGER.info(f'{id2type[int(item[5])]}:{pos}    {item[4]}')
                LOGGER.info(f'---------------------{im0.shape}')

        return json.dumps(res_list)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default=ROOT / 'runs/server', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def delegate(method, args):
    # define remote methods:
    if method == 'echo':
        return args
    elif method == 'load_model':
        if detector:
            detector.model = Detector.load_model(**args)
        else:
            raise RuntimeError("an server error: detector not found")
    elif method == 'detect':
        if detector:
            print(args)
            return detector.run(**args)
        else:
            raise RuntimeError("an server error: detector not found")
    else:
        raise RuntimeError("an server error")


def main(port=8888):
    opt = parse_opt()
    global detector
    detector = Detector(**vars(opt))  # load default model at initial
    s = procbridge.Server('0.0.0.0', port, delegate)
    s.start(daemon=False)
    print(f'Server is on {port}...')


if __name__ == "__main__":
    main()
