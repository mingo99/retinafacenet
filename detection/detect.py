import torch
import numpy as np
import cv2
import time
from PIL import Image
from model import retinafacenet_resnet50_fpn_with_weight
from ._utils import predict, draw_boxes

def detect_image(input, threshold, path=None):
    """
    Test a image.

    Arguments:
        input(String): Path of image
        threshold(Float): The threshold of scores to save predict results
        quantize(Bool): Indicate whether to quantize model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = retinafacenet_resnet50_fpn_with_weight(path)
    model.eval().to(device)
    # read the image
    image = Image.open(input)
    # detect outputs
    boxes, keyps, classes, scores, labels = predict(image, model, device, threshold)
    # draw bounding boxes
    image = draw_boxes(boxes, keyps, classes, scores, labels, image)
    save_name = f"{input.split('/')[-1].split('.')[0]}_{''.join(str(threshold).split('.'))}"
    cv2.imwrite(f"samples/outputs/{save_name}.jpg", image)
    cv2.imshow('Image', image)
    cv2.waitKey(0)

def detect_video(input, threshold, path):
    """
    Test a video, single step is same as image.

    Arguments:
        input(String): Path of video
        threshold(Float): The threshold of scores to save predict results
        quantize(Bool): Indicate whether to quantize model
    """
    # define the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = retinafacenet_resnet50_fpn_with_weight(path)
    model.eval().to(device)

    cap = cv2.VideoCapture(input)
    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')
    # get the frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    save_name = f"{input.split('/')[-1].split('.')[0]}_{''.join(str(threshold).split('.'))}"
    # define codec and create VideoWriter object 
    out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                        cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                        (frame_width, frame_height))
    frame_count = 0 # to count total frames
    total_fps = 0 # to get the final frames per second

    # read until end of video
    while(cap.isOpened()):
        # capture each frame of the video
        ret, frame = cap.read()
        if ret == True:
            # get the start time
            start_time = time.time()
            with torch.no_grad():
                # get predictions for the current frame
                boxes, classes, scores, labels = predict(frame, model, device, threshold)
            # draw boxes and show current frame on screen
            image = draw_boxes(boxes, classes, scores, labels, frame)
            # get the end time
            end_time = time.time()
            # get the fps
            fps = 1 / (end_time - start_time)
            # add fps to total fps
            total_fps += fps
            # increment frame count
            frame_count += 1
            # write the FPS on the current frame
            cv2.putText(image, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            # convert from BGR to RGB color format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('image', image)
            out.write(image)
            # press `q` to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # release VideoCapture()
    cap.release()
    # close all frames and video windows
    cv2.destroyAllWindows()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")