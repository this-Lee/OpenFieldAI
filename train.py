import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

# HybridNets 전용 모듈 (기존 소스코드에 포함된 파일들)
from model.hybridnets import HybridNet
from utils.utils import CustomDataset, collate_fn, get_optimizer, get_scheduler
from utils.loss import FocalLoss, SegmentationLoss

def parse_args():
    parser = argparse.ArgumentParser(description='HybridNets Training for Open Field')
    parser.add_argument('-p', '--project', type=str, default='openfield', help='project yaml file name')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientnet backbone')
    parser.add_argument('-n', '--num_workers', type=int, default=4, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=8, help='total batch size for all gpus')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_interval', type=int, default=500, help='interval to save model')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('--saved_path', type=str, default='weights/')
    args = parser.parse_args()
    return args

def train():
    args = parse_args()
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.saved_path, exist_ok=True)

    # 1. 데이터셋 및 데이터로더 설정
    # 앞서 분할한 train/val 경로를 자동으로 읽어옵니다.
    train_dataset = CustomDataset(split='train', project_name=args.project)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 2. 모델 설정 (HybridNets)
    model = HybridNet(compound_coef=args.compound_coef, num_classes=train_dataset.num_classes)
    model = model.cuda()
    model.train()

    # 3. 손실 함수 및 최적화 도구
    criterion_det = FocalLoss() # Detection용 Focal Loss
    criterion_seg = SegmentationLoss() # Segmentation용 Loss (Dice + BCE)
    optimizer = get_optimizer(model, args.lr)
    scheduler = get_scheduler(optimizer)
    writer = SummaryWriter(args.log_path)

    print(f"✅ 학습 시작: 총 {len(train_loader)} 배치 / {args.num_epochs} 에포크")

    step = 0
    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for i, (images, annots, masks) in enumerate(pbar):
            images = images.cuda().float()
            annots = annots.cuda()
            masks = masks.cuda()

            optimizer.zero_grad()

            # 모델 추론
            features, regressions, classifications, anchors, seg_out = model(images)

            # 손실 계산
            loss_det, cls_loss, reg_loss = criterion_det(classifications, regressions, anchors, annots)
            loss_seg = criterion_seg(seg_out, masks)

            # 멀티태스크 가중치 조절 (필요 시 조절 가능)
            total_loss = loss_det + loss_seg

            # 역전파 및 최적화
            total_loss.backward()
            optimizer.step()

            step += 1

            # 로그 기록
            pbar.set_postfix({
                'Total': f"{total_loss.item():.4f}",
                'Det': f"{loss_det.item():.4f}",
                'Seg': f"{loss_seg.item():.4f}"
            })

            writer.add_scalar('Loss/Total', total_loss.item(), step)
            writer.add_scalar('Loss/Detection', loss_det.item(), step)
            writer.add_scalar('Loss/Segmentation', loss_seg.item(), step)

            # 모델 저장
            if step % args.save_interval == 0:
                save_name = f"hybridnets_{args.project}_{epoch}_{step}.pth"
                torch.save(model.state_dict(), os.path.join(args.saved_path, save_name))

        scheduler.step()

    writer.close()
    print("✨ 모든 학습이 완료되었습니다!")

if __name__ == '__main__':
    train()