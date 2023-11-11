import cv2
import numpy as np
import torch
from skimage import transform as trans

from datasets import COCOFB_INSTANCE_CATEGORY_NAMES as coco_names

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


def estimate_norm(lmk):
    """Transform"""
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    src = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    tform.estimate(lmk, src)
    normer = tform.params[0:2, :]
    return normer


def norm_crop(img, landmark, image_size=112):
    """Crop face"""
    normer = estimate_norm(landmark)
    warped = cv2.warpAffine(
        img, normer, (image_size, image_size), borderValue=[0.0, 0.0, 0.0, 0.0]
    )
    return warped


def pre_process(image, landmarks):
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    imgs = []
    for landmark in landmarks:
        landmark = np.array(landmark, dtype="float32")
        img = norm_crop(image, landmark)
        imgs.append(img)
    return imgs


def post_process(imgs):
    imgs1 = []
    for img in imgs:
        img = img.transpose((2, 0, 1))
        img = (img - 127.5) / 127.5
        imgs1.append(img)
    return imgs1


def infer(imgs, model, device):
    """
    Extract image features
    """
    if len(imgs.shape) == 3:
        imgs = imgs[np.newaxis, :]
    features = []
    for i in range(imgs.shape[0]):
        img = imgs[i][np.newaxis, :]
        img = torch.tensor(img, dtype=torch.float32, device=device)
        feature = model(img)
        feature = feature.detach().cpu().numpy()
        features.append(feature)
    return features


def draw_boxes(image, boxes_face, boxes_body, names):
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    if boxes_face is not None:
        for i in range(boxes_face.shape[0]):
            bbox_face = boxes_face[i, :4]
            bbox_body = boxes_body[i]
            name = names[i]
            # 画人脸框
            cv2.rectangle(
                img,
                (int(bbox_face[0]), int(bbox_face[1])),
                (int(bbox_face[2]), int(bbox_face[3])),
                (255, 0, 0),
                1,
            )
            cv2.putText(
                image,
                name,
                (int(bbox_face[0]), int(bbox_face[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                lineType=cv2.LINE_AA,
            )
            # 画人体框
            cv2.rectangle(
                img,
                (int(bbox_body[0]), int(bbox_body[1])),
                (int(bbox_body[2]), int(bbox_body[3])),
                (0, 255, 0),
                1,
            )
            cv2.putText(
                image,
                name,
                (int(bbox_body[0]), int(bbox_body[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                lineType=cv2.LINE_AA,
            )
    return img
