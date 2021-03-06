import os
import sys
from pathlib import Path

import imutils

import opt
import time
import cv2
import torch
import numpy as np

from Wang.object_counting import object_counting_down_d5

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend, letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from utils.general import xyxy2xywh
from Wang.check_slot import spot_dict_d3
from Wang.draw_polygon import *
from Wang.object_counting import object_counting_up_d3


def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


cfg = get_config()
cfg.merge_from_file(opt.config_deepsort)
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

source = str(opt.source)
# Load model
print(opt.weights)
device = select_device(opt.device)
model = DetectMultiBackend(opt.weights, device=device, dnn=opt.dnn)
stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
imgsz = check_img_size(opt.imgsz, s=stride)  # check image size

# Half
half = device.type != 'cpu'  # half precision only supported on CUDA
if half:
    model.model.half() if opt.half else model.model.float()

dt, seen = [0.0, 0.0, 0.0], 0
check = -1


def detect(img0):
    im0 = img0.copy()
    img = letterbox(img0, opt.imgsz, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(img)
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    pred = model(im, augment=opt.augment, visualize=opt.visualize)

    # NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        spot_dict = {}
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            xyxys, confs, clss = [], [], []
            for *xyxy, conf, cls in reversed(det):
                label = '%s %.2f' % (names[int(cls)], conf)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                xyxys.append([x1, y1, x2, y2])
                confs.append(conf)
                clss.append(cls)
                # cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            xywhs = xyxy2xywh(torch.Tensor(xyxys))
            confs = torch.Tensor(confs)
            clss = torch.tensor(clss)
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):
                    x1, y1, x2, y2 = output[0:4]
                    if y2 < 58:
                        continue
                    # x_center = (x1 + x2) // 2
                    # y_center = (y1 + y2) // 2
                    id = output[4] + 300
                    # id_dict[id] = [x_center, y_center]
                    spot_dict[id] = [x1, y1, x2, y2, id]
                    # # cls = output[5]
                    # c = int(cls)  # integer class
                    # color = compute_color_for_id(id)
                    # cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
                    # cv2.putText(im0, str(id), (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
    try:
        return im0, spot_dict
    except:
        return img0, None


if __name__ == '__main__':
    path = r"D:\cam_thu_vien\17_3_2022\D3\2022-03-17\D3_cut.mp4"
    cap = cv2.VideoCapture(path)
    # fourcc = 'mp4v'  # output video codec
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # vid_writer = cv2.VideoWriter(r"D:\Lab IC\demo\ch16_C3 v??o_sau 17h25 06012022.mp4", cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    check = 1
    rotate = 0
    frame = 0
    skip = 10  # seconds
    old_dict = {}
    lp_dict = {}
    old_key_lp = []
    old_key_d5 = []
    count_missing_tracks = 0
    final_dict = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    with torch.no_grad():
        while True:
            frame += 1
            if frame % 5 != 0:
                continue
            spot_dict_d3_copy = spot_dict_d3.copy()
            t = time.time()
            ret, img0 = cap.read()
            img0 = cv2.resize(img0, dsize=(1280, 720))
            cv2.imwrite("main/D3_original.jpg", img0)
            H, W = img0.shape[:2]
            img0, spot_dict = detect(img0)
            alpha = 0.7
            img0_copy = img0.copy()
            new_key = list(set(spot_dict.keys()) - set(old_dict.keys()))
            key_up = object_counting_up_d3(old_dict, spot_dict)
            print(key_up)
            if len(new_key) == 1:
                x1, y1, x2, y2, id = spot_dict[new_key[0]]
                if y2 > H - 100:
                    old_key_lp = new_key
                elif H - 100 > y2 > 58:
                    old_key_d5 = new_key
            old_dict = spot_dict
            for key in lp_dict.keys():
                if key not in spot_dict.keys():
                    count_missing_tracks += 1
                    break
            if count_missing_tracks > 25 * 3:
                for key in lp_dict.keys():
                    if key not in spot_dict.keys():
                        del lp_dict[key]
                        break
            if key_up is not None:
                if key_up in lp_dict.keys():
                    with open(fr"D3_up/{lp_dict[key_up]}", "w+") as f:
                        pass
            if len(old_key_lp) == 1:
                if len(os.listdir("LP")) > 0:
                    lp_text = os.listdir("LP")[0]
                    os.remove("LP/" + lp_text)
                    lp_dict[id] = lp_text
                    old_key_lp = []
            if len(old_key_d5) == 1:
                if len(os.listdir("D5_down")) > 0:
                    d5_text = os.listdir("D5_down")[0]
                    os.remove("D5/" + lp_text)
                    lp_dict[id] = d5_text
                    old_key_d5 = []
            for key, value in spot_dict.items():
                x1, y1, x2, y2, id = value
                x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
                if id in lp_dict.keys():
                    id = lp_dict[id]
                for k, v in spot_dict_d3_copy.items():
                    x, y, w, h, status, _ = v
                    if x < x_center < x + w and y < y_center < y + h:
                        spot_dict_d3_copy[k] = [x, y, w, h, 1, id]

                if len(str(id)) < 6:
                    cv2.rectangle(img0, (x1, y1 - 5), (x1 + 65, y1 + 25), (0, 0, 0), -1)
                    cv2.putText(img0, str(str(id).upper()), (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (255, 255, 255), 2)
                    continue
                cv2.rectangle(img0, (x1, y1 - 5), (x1 + 200, y1 + 25), (0, 0, 0), -1)
                cv2.putText(img0, str(str(id).upper()), (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (255, 255, 255), 2)
            f = open("main/D3_result.txt", "w+")
            for k, v in spot_dict_d3_copy.items():
                x, y, w, h, status, id = v
                if len(str(id)) < 6:
                    id = ""
                f.write(f"{k} 2 {status} {id}\n")
                if status == 0:
                    draw_polygon_d3(img0_copy, k, color=(0, 255, 0))
                else:
                    draw_polygon_d3(img0_copy, k, color=(0, 0, 255))
            f.close()
            img0 = cv2.addWeighted(img0, alpha, img0_copy, 1 - alpha, 0)
            cv2.imwrite("main/D3.jpg", img0)
            cv2.imshow("Image", imutils.resize(img0, width=960))
            # vid_writer.write(img0)
            key = cv2.waitKey(1)
            print("FPS: ", 1 // (time.time() - t))
            if key == ord("q"):
                break
            if key == ord("c"):
                check = -check
            if key == ord("r"):
                rotate += 1
            if key == ord("n"):
                frame += skip * 25
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            elif key == ord("p") and frame > skip * 25:
                frame -= skip * 25
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            elif key == 32:
                cv2.waitKey()
