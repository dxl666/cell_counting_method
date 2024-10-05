import os
import time
import datetime

import numpy as np
import torch
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import BaseDataset
import transforms as T
from unet import UNet
from da_transUNet import DATransUNet
from model1 import CONFIGS as CONFIGS_ViT_seg
from saunet import SAUNet
from cell_unet import CellUNet
from c_resunet import CResUNet
from model1 import SCTransUNet
from model2 import SCTransUNet2


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            # T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return SegmentationPresetEval(mean=mean, std=std)


def create_config(model_name):
    if model_name == "SCTransUNet" or model_name == "DATransUNet" or model_name == "SCTransUNet2":
        vit_config = CONFIGS_ViT_seg['R50-ViT-B_16']
        vit_config.n_skip = 3
        vit_config.patches.grid = (14, 14)
    else:
        vit_config = None
    return vit_config


def create_model(num_classes, model_name):
    config_vit = create_config(model_name)
    if model_name == "UNet":
        model = UNet(in_channels=3, base_dim=64, num_classes=num_classes, bilinear=False, mode='conv')
    elif model_name == "SAUNet":
        model = SAUNet(in_channels=3, base_dim=64, num_classes=num_classes, self_attn=True)
    elif model_name == "CellUNet":
        model = CellUNet(in_channels=3, base_dim=64, num_classes=num_classes)
    elif model_name == "CResUNet":
        model = CResUNet(in_channels=3, base_dim=16, num_classes=num_classes, mode='maxpool')
    elif model_name == "DATransUNet":
        model = DATransUNet(config_vit, 224, num_classes)
    elif model_name == "SCTransUNet":
        model = SCTransUNet(config_vit, 224, num_classes)
    elif model_name == "SCTransUNet2":
        model = SCTransUNet2(config_vit, 224, num_classes)
    else:
        raise ValueError("model name is error!")
    if config_vit is not None:
        config_vit.n_classes = num_classes
        model.load_from(weights=np.load(config_vit.pretrained_path))
    return model


def test(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # classes + background
    class_nums = args.num_classes + 1
    if args.dataset == "A549":
        mean = (0.604, 0.604, 0.603)
        std = (0.055, 0.054, 0.058)
    elif args.dataset == "liveCell":
        mean = (0.502, 0.502, 0.502)
        std = (0.039, 0.039, 0.039)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    # test_dataset = BaseDataset(args.data_path, split='test', transform=get_transform(mean=mean, std=std),
    #                            enhance=False)
    # enhance_test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                                   batch_size=1,
    #                                                   num_workers=0,
    #                                                   pin_memory=True,
    #                                                   collate_fn=test_dataset.collate_fn)
    enhance_test_dataset = BaseDataset(args.data_path, split='test', transform=get_transform(mean=mean, std=std),
                                       enhance=True)

    enhance_test_loader = torch.utils.data.DataLoader(enhance_test_dataset,
                                                      batch_size=1,
                                                      num_workers=0,
                                                      pin_memory=True,
                                                      collate_fn=enhance_test_dataset.collate_fn)
    # model_list = ['UNet', 'SAUNet', 'CellUNet', 'CResUNet', 'SCTransUNet']
    # model_list = ['UNet', 'SAUNet', 'CResUNet', 'SCTransUNet', 'DATransUNet']
    model_list = ['SCTransUNet2']
    for model_name in model_list:
        # last_weight_path = './save_weights/' + args.dataset + '_' + model_name + '.pth'
        # best_weight_path = './save_weights/' + args.dataset + '_best' + model_name + '.pth'
        # 2: enhance
        last_weight_path = './save_weights/' + args.dataset + '_' + model_name + '_enhance.pth'
        best_weight_path = './save_weights/' + args.dataset + '_best' + model_name + '_enhance.pth'
        model = create_model(num_classes=class_nums, model_name=model_name)
        model.load_state_dict(
            torch.load(best_weight_path if os.path.exists(best_weight_path) else last_weight_path,
                       map_location='cpu')[
                'model'])
        model.to(device)

        confmat, dice, _, accuracy, precise, recall, f1 = evaluate(model,
                                                                   enhance_test_loader,
                                                                   device, class_nums)
        mIoU = confmat.return_mIoU()
        accuracy = accuracy * 100
        precise = precise * 100
        recall = recall * 100
        f1 = f1 * 100
        dice = dice * 100
        mIoU = mIoU * 100
        enhance = 'enhance '
        print(enhance + args.dataset + ' dataset: ' + model_name)
        print(f'mIoU: {mIoU:.2f}')
        print(f'dice coefficient: {dice:.2f}')
        print(f'counting accuracy: {accuracy:.2f}, '
              f'precise: {precise:.2f}, '
              f'recall: {recall:.2f}, f1: {f1:.2f}')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda:1", help="training device")
    parser.add_argument("--dataset", default="MBM", type=str,
                        help="the current dataset's name")
    parser.add_argument("--data-path", default="./DRIVE224_2", help="DRIVE root")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    test(args)
