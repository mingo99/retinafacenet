import os
import time

import cv2 as cv
import numpy as np
import torch
from PIL import Image

import models
from detection import infer as infer_det

from ._utils import draw_boxes, infer, post_process, pre_process


def load_face_db(face_db_path, det_model, rec_model, device):
    faces_db = {}
    for path in os.listdir(face_db_path):
        name = os.path.basename(path).split(".")[0]
        image_path = os.path.join(face_db_path, path)
        img = Image.open(image_path)
        _, keyps, _, _, labels = infer_det(img, det_model, device, 0.5)
        landmarks = keyps[labels == 2]
        imgs = pre_process(img, landmarks)
        if imgs is None or len(imgs) > 1:
            print("人脸库中的 %s 图片包含不是1张人脸，自动跳过该图片" % image_path)
            continue
        imgs = post_process(imgs)
        feature = infer(imgs[0], rec_model, device)
        faces_db[name] = feature[0][0]
    return faces_db


def match_face(faces_db, images, landmarks, rec_model, threshold, device):
    imgs = pre_process(images, landmarks)
    if imgs is None:
        return None
    faces = post_process(imgs)
    # imgs = np.array(imgs, dtype="float32")
    s = time.time()
    features = infer(faces, rec_model, device)
    print("人脸识别时间：%dms" % int((time.time() - s) * 1000))
    names = []
    probs = []
    for i in range(len(features)):
        feature = features[i][0]
        results_dict = {}
        for name in faces_db.keys():
            feature1 = faces_db[name]
            prob = np.dot(feature, feature1) / (
                np.linalg.norm(feature) * np.linalg.norm(feature1)
            )
            results_dict[name] = prob
        results = sorted(results_dict.items(), key=lambda d: d[1], reverse=True)
        print("人脸对比结果：", results)
        result = results[0]
        prob = float(result[1])
        probs.append(prob)
        if prob > threshold:
            name = result[0]
            names.append(name)
        else:
            names.append("unknow")
    return names


def match_fb(face_boxes, body_boxes):
    match_boxes = []
    match_ids = []
    for _, fb in enumerate(face_boxes):
        bodys = []
        ids = []
        for id, bb in enumerate(body_boxes):
            if fb[0] < bb[0]:
                continue
            if fb[1] < bb[1]:
                continue
            if fb[2] > bb[2]:
                continue
            if fb[3] > bb[3]:
                continue
            bodys.append(bb)
            ids.append(id)
        if len(bodys) > 1:
            x_dist = []
            for bb in bodys:
                body_x_center = (bb[0] + bb[2]) / 2
                face_x_center = (fb[0] + fb[2]) / 2
                x_dist.append(abs(body_x_center - face_x_center))
            idx = x_dist.index(min(x_dist))
            match_boxes.append(bodys[idx])
            match_ids.append(ids[idx])
        elif len(bodys) == 1:
            match_boxes.append(bodys[0])
            match_ids.append(ids[0])
        else:
            match_boxes.append([0, 0, 0, 0])
            match_ids.append("Nan")
    return match_boxes, match_ids


def recognize(faces_db_path, image_path, weights, thresholds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det_model = models.get_model("retinafacenet_resnet50_fpn", weights=weights[0])
    det_model.eval().to(device)
    rec_model = models.get_model("facenet_mobilev2", weights=weights[1])
    rec_model.eval().to(device)

    faces_db = load_face_db(faces_db_path, det_model, rec_model, device)

    image = Image.open(image_path)
    boxes, keyps, _, _, labels = infer_det(image, det_model, device, thresholds[0])
    face_boxes = boxes[labels == 2]
    body_boxes = boxes[labels == 1]
    landmarks = keyps[labels == 2]

    names = match_face(faces_db, image, landmarks, rec_model, thresholds[1], device)
    match_body_boxes, _ = match_fb(face_boxes, body_boxes)

    image = draw_boxes(image, face_boxes, match_body_boxes, names)
    cv.imwrite("result.jpg", image)
    cv.imshow("Image", image)
    cv.waitKey(0)
