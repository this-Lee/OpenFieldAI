import argparse
import datetime
import os
import traceback
import sys
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torchvision import transforms
from tqdm import tqdm

# 깃허브 원본 구조 임포트
from val import val
from backbone import HybridNetsBackbone
from utils.utils import get_last_weights, init_weights, boolean_string, \
save_checkpoint, DataLoaderX, Params
from hybridnets.dataset import BddDataset
from hybridnets.custom_dataset import CustomDataset
from hybridnets.autoanchor import run_anchor
from hybridnets.model import ModelWithLoss
from utils.constants import *
from collections import OrderedDict

def get_args():
    parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - OpenField')
    parser.add_argument('-p', '--project', type=str, default='openfield', help='Project yaml file')
    parser.add_argument('-bb', '--backbone', type=str, default='efficientnet-b3')
    parser.add_argument('-c', '--compound_coef', type=int, default=3)
    parser.add_argument('-n', '--num_workers', type=int, default=8)
    parser.add_argument('-b', '--batch_size', type=int, default=8) # 메모리에 따라 조절
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--data_path', type=str, default='datasets/', help='Dataset root')
    parser.add_argument('--log_path', type=str, default='checkpoints/')
    parser.add_argument('--saved_path', type=str, default='checkpoints/')
    parser.add_argument('--load_weights', type=str, default=None)
    parser.add_argument('--amp', type=boolean_string, default=True) # 속도 향상을 위한 AMP
    args = parser.parse_args()
    return args

def train(opt):
    torch.backends.cudnn.benchmark = True
    params = Params(f'projects/{opt.project}.yml')

    # 가중치 및 로그 저장 경로 설정
    opt.saved_path = opt.saved_path + f'/{opt.project}/'
    opt.log_path = opt.log_path + f'/{opt.project}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    seg_mode = BINARY_MODE # 노지 주행 영역은 보통 단일 클래스(도로)

    # 1. 커스텀 데이터셋 설정 (NIA 데이터셋용)
    train_dataset = CustomDataset(
        params=params,
        is_train=True,
        inputsize=params.model['image_size'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=params.mean, std=params.std)
        ]),
        seg_mode=seg_mode
    )

    training_generator = DataLoaderX(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=params.pin_memory,
        collate_fn=CustomDataset.collate_fn
    )

    # 2. 모델 초기화 (HybridNetsBackbone 사용)
    model = HybridNetsBackbone(
        num_classes=len(params.obj_list),
        compound_coef=opt.compound_coef,
        ratios=eval(params.anchors_ratios),
        scales=eval(params.anchors_scales),
        seg_classes=len(params.seg_list),
        backbone_name=opt.backbone,
        seg_mode=seg_mode
    )

    if opt.load_weights:
        model.load_state_dict(torch.load(opt.load_weights).get('model', {}), strict=False)
    else:
        init_weights(model)

    # Loss 계산을 포함한 모델 래핑
    model = ModelWithLoss(model, debug=False)
    model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)
    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    step = 0
    model.train()

    try:
        for epoch in range(opt.num_epochs):
            epoch_loss = []
            progress_bar = tqdm(training_generator, ascii=True)
            for iter, data in enumerate(progress_bar):
                imgs = data['img'].cuda().to(memory_format=torch.channels_last)
                annot = data['annot'].cuda()
                seg_annot = data['segmentation'].cuda()

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=opt.amp):
                    cls_loss, reg_loss, seg_loss, _, _, _, _ = model(imgs, annot, seg_annot, obj_list=params.obj_list)
                    loss = cls_loss.mean() + reg_loss.mean() + seg_loss.mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss.append(loss.item())
                step += 1

                progress_bar.set_description(f'Epoch: {epoch}/{opt.num_epochs}. Loss: {loss.item():.4f}')
                writer.add_scalar('Loss/Total', loss.item(), step)

            scheduler.step(np.mean(epoch_loss))
            save_checkpoint(model, opt.saved_path, f'hybridnets_epoch_{epoch}.pth')

    except KeyboardInterrupt:
        save_checkpoint(model, opt.saved_path, 'interrupted.pth')
    finally:
        writer.close()

if __name__ == '__main__':
    opt = get_args()
    train(opt)