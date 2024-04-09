import argparse
import os
import time

import cv2 as cv
import torch
from PIL import Image

import models
from detection import detect_image, detect_video, draw_boxes
from detection import infer as det_infer
from recognition import recognize


def crop_body():
    device = torch.device("cpu")
    model = models.get_model("retinafacenet_resnet50_fpn", weights="./weights/retinafacenet_resnet50.pth")
    model.eval().to(device)

    folder = "../../data/test/fmg/"
    # names = os.listdir(folder)
    names = ["19.jpg", "20.jpg", "21.jpg"]
    for name in names:
        file_path = os.path.join(folder, name)
        image = Image.open(file_path)
        image_raw = cv.imread(file_path)
        # detect outputs
        boxes, keyps, classes, scores, labels = det_infer(image, model, device, 0.5)
        bodys = []
        for box in boxes:
            x1, y1, x2, y2 = box
            body = image_raw[y1:y2, x1:x2]
            new_size = (64,128)
            body = cv.resize(body, new_size)
            bodys.append(body)
            cv.imwrite(f"../../data/MyMarket/gallery/03/{time.time()}.jpg", body)
        cv.imshow("Image", bodys[0])
        cv.waitKey(0)

def simple_test():
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--video",
        default=False,
        type=bool,
        help="indicate test whether video or image",
    )
    parser.add_argument(
        "-i",
        "--input",
        default="../../data/test/fmg/2.jpg",
        help="path to input input image",
    )
    parser.add_argument(
        "-t", "--threshold", default=0.5, type=float, help="detection threshold"
    )
    parser.add_argument(
        "-q", "--quantize", default=False, type=bool, help="whether to quantize model"
    )
    parser.add_argument(
        "-p",
        "--path",
        default="./weights/retinafacenet_resnet50.pth",
        type=str,
        help="Path of trained model.",
    )
    args = vars(parser.parse_args())

    device = torch.device("cpu")
    model = models.get_model("retinafacenet_resnet50_fpn", weights="./weights/retinafacenet_resnet50.pth")
    model.eval().to(device)

    if args["video"]:
        detect_video(args["input"], args["threshold"], args["path"])
    else:
        detect_image(model,device,args["input"], args["threshold"], args["path"])


def infer():
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="../../data/test/9.jpg",
        help="path to input input image",
    )
    args = vars(parser.parse_args())

    weights = [
        "./weights/retinafacenet_resnet50.pth",
        "./weights/facenet_mobilev2.pth",
    ]
    thres = [0.5, 0.3]

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    det_model = models.get_model("retinafacenet_resnet50_fpn", weights=weights[0])
    det_model.eval().to(device)
    rec_model = models.get_model("facenet_mobilev2", weights=weights[1])
    rec_model.eval().to(device)

    # for i in range(10):
    #     folder = "../../data/test/"
    #     recognize(det_model,rec_model,device,"./face_db", f"{folder}{10-i}.jpg", weights, thres)

    recognize(det_model,rec_model,device,"./face_db", args["input"], weights, thres)

def export_weights():
    rec_model = models.get_model(
        "facenet_mobilev2", weights="./weights/facenet_mobilev2.pth"
    )
    torch.save(rec_model.state_dict(), "./weights/facenet_mobilev2_state_dict.pth")

def fps_test():
    folder = "../../data/test/"
    device = torch.device("cpu")
    model = models.get_model("retinafacenet_resnet50_fpn", weights="./weights/retinafacenet_resnet50.pth")
    model.eval().to(device)

    for i in range(10):
        detect_image(model,device,f"{folder}{10-i}.jpg", 0.5, "./weights/retinafacenet_resnet50.pth")

if __name__ == "__main__":
    # export_weights()
    # simple_test()
    # infer()
    crop_body()
