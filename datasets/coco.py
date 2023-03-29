import os

import torch
import torchvision
from . import transforms as T
from . import presets
from . import utils

from .group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def get_coco(root, image_set, year, transforms, mode="instances"):
    anno_file_template = "{}_{}{}.json"
    PATHS = {
        "train": (f"images/train{year}", os.path.join("annotations", anno_file_template.format(mode, "train", year))),
        "val": (f"images/val{year}", os.path.join("annotations", anno_file_template.format(mode, "val", year))),
        # "train": ("val2014", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    t = [utils.ConvertCocoPolysToMask()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    #TODO Modified by Mingo
    # if image_set == "train":
    #     dataset = utils._coco_remove_images_without_annotations(dataset)

    # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])

    return dataset


def get_coco_kp(root, image_set, year, transforms):
    return get_coco(root, image_set, year, transforms, mode="person_keypoints")


def get_dataset(name, image_set, year, transform, data_path, mode):
    paths = {"coco": (data_path, get_coco, 3), "coco_kp": (data_path, get_coco_kp, 2)}
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, year=year, transforms=transform, mode=mode)
    return ds, num_classes


def get_transform(train, args):
    if train:
        return presets.DetectionPresetTrain(data_augmentation=args.data_augmentation)
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval()


def get_dataloader(args):
    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(args.dataset, "train", 2023, get_transform(True, args), args.data_path, "keypoints")
    dataset_test, _ = get_dataset(args.dataset, "val", 2023, get_transform(False, args), args.data_path, "keypoints")

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_collate_fn = utils.collate_fn
    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")

        train_collate_fn = utils.copypaste_collate_fn

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=train_collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    return data_loader, data_loader_test, num_classes, train_sampler