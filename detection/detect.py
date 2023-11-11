import time

import cv2 as cv
import torch
from PIL import Image

import models

from ._utils import draw_boxes, infer


def detect_image(image_in, threshold, weights=None):
    """
    Test a image.

    Arguments:
        input(Str): Path of image
        threshold(Float): The threshold of scores to save predict results
        weights(Str): Weights path of model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.get_model("retinafacenet_resnet50_fpn", weights=weights)
    model.eval().to(device)
    # read the image
    image = Image.open(image_in)
    # detect outputs
    boxes, keyps, classes, scores, labels = infer(image, model, device, threshold)
    # draw bounding boxes
    image = draw_boxes(boxes, keyps, classes, scores, labels, image)
    save_name = (
        f"{image_in.split('/')[-1].split('.')[0]}_{''.join(str(threshold).split('.'))}"
    )
    cv.imwrite(f"samples/outputs/{save_name}.jpg", image)
    cv.imshow("Image", image)
    cv.waitKey(0)


def detect_video(video_in, threshold, weights):
    """
    Test a video, single step is same as image.

    Arguments:
        video_in(String): Path of video
        threshold(Float): The threshold of scores to save predict results
        quantize(Bool): Indicate whether to quantize model
    """
    # define the computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.get_model("retinafacenet_resnet50_fpn", weights=weights)
    model.eval().to(device)

    cap = cv.VideoCapture(video_in)
    if cap.isOpened():
        print("Error while trying to read video. Please check path again")
    # get the frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    save_name = (
        f"{video_in.split('/')[-1].split('.')[0]}_{''.join(str(threshold).split('.'))}"
    )
    # define codec and create VideoWriter object
    out = cv.VideoWriter(
        f"outputs/{save_name}.mp4",
        cv.VideoWriter.fourcc(*"mp4v"),
        30,
        (frame_width, frame_height),
    )
    frame_count = 0  # to count total frames
    total_fps = 0  # to get the final frames per second

    # read until end of video
    while cap.isOpened():
        # capture each frame of the video
        ret, frame = cap.read()
        if ret:
            # get the start time
            start_time = time.time()
            with torch.no_grad():
                # get predictions for the current frame
                boxes, keyps, classes, scores, labels = infer(
                    frame, model, device, threshold
                )
            # draw boxes and show current frame on screen
            image = draw_boxes(boxes, keyps, classes, scores, labels, frame)
            # get the end time
            end_time = time.time()
            # get the fps
            fps = 1 / (end_time - start_time)
            # add fps to total fps
            total_fps += fps
            # increment frame count
            frame_count += 1
            # write the FPS on the current frame
            cv.putText(
                image,
                f"{fps:.3f} FPS",
                (15, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            # convert from BGR to RGB color format
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            cv.imshow("image", image)
            out.write(image)
            # press `q` to exit
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    # release VideoCapture()
    cap.release()
    # close all frames and video windows
    cv.destroyAllWindows()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
