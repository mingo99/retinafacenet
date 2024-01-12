import argparse

import torch

import models
from detection import detect_image, detect_video
from recognition import recognize


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
        default="../../data/test/2.jpg",
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
        default="./checkpoints/server/checkpoint.pth",
        type=str,
        help="Path of trained model.",
    )
    args = vars(parser.parse_args())

    if args["video"]:
        detect_video(args["input"], args["threshold"], args["path"])
    else:
        detect_image(args["input"], args["threshold"], args["path"])


def infer():
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="../../data/test/2.jpg",
        help="path to input input image",
    )
    args = vars(parser.parse_args())

    weights = [
        "./weights/retinafacenet_resnet50.pth",
        "./weights/facenet_mobilev2_state_dict.pth",
    ]
    thres = [0.5, 0.5]
    recognize("./face_db", args["input"], weights, thres)


def export_weights():
    rec_model = models.get_model(
        "facenet_mobilev2", weights="./weights/facenet_mobilev2.pth"
    )
    torch.save(rec_model.state_dict(), "./weights/facenet_mobilev2_state_dict.pth")


if __name__ == "__main__":
    # simple_test()
    infer()
