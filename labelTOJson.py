import json
import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# 1. 경로 설정
RAW_DATA_DIR = './raw_data'      # 시퀀스 폴더들이 있는 곳
OUTPUT_ROOT = './datasets/field_data'
VAL_SIZE = 0.2                   # 20%의 시퀀스를 검증셋으로 사용
IMG_W, IMG_H = 1920, 1080

# 클래스 매핑 (이전 대화 기준)
DET_CLASS_MAP = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}

def convert_bbox(bbox, w, h):
    x, y, bw, bh = bbox
    return [(x + bw/2)/w, (y + bh/2)/h, bw/w, bh/h]

def process_sequence(seq_path, split_type):
    """특정 시퀀스 폴더를 처리하여 train 또는 val 폴더로 저장"""
    json_path = list(Path(seq_path).glob('*.json'))[0] # 폴더 내 json 찾기
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 출력 경로 설정
    img_out = os.path.join(OUTPUT_ROOT, split_type, 'images')
    lab_out = os.path.join(OUTPUT_ROOT, split_type, 'labels')
    seg_out = os.path.join(OUTPUT_ROOT, split_type, 'segmentation')

    for folder in [img_out, lab_out, seg_out]: os.makedirs(folder, exist_ok=True)

    # 이미지 메타데이터 매핑
    img_info_map = {img['id']: img for img in data.get('images', [])}

    # 1. Bbox 처리
    annotations = data.get('annotations', [])
    img_labels = {img_id: [] for img_id in img_info_map.keys()}
    for ann in annotations:
        cat_id = ann['category_id']
        if cat_id in DET_CLASS_MAP:
            yolo_box = convert_bbox(ann['bbox'], IMG_W, IMG_H)
            label_str = f"{DET_CLASS_MAP[cat_id]} " + " ".join([f"{v:.6f}" for v in yolo_box])
            img_labels[ann['image_id']].append(label_str)

    # 2. 이미지 및 세그멘테이션 처리
    for img_id, info in img_info_map.items():
        file_name = info['file_name']
        base_name = Path(file_name).stem

        # 원본 이미지 복사
        src_img_path = os.path.join(seq_path, 'images', file_name)
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, os.path.join(img_out, file_name))

        # YOLO 라벨 저장
        with open(os.path.join(lab_out, f"{base_name}.txt"), 'w') as f:
            f.write("\n".join(img_labels[img_id]))

        # 세그멘테이션 마스크 생성 (Polygon 데이터가 있을 경우)
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        if 'objects' in info: # Polygon 데이터 구조일 때
            for obj in info['objects']:
                if obj['label'] == 'common_road':
                    poly = np.array(obj['position']).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 1)
        cv2.imwrite(os.path.join(seg_out, f"{base_name}.png"), mask)

# 메인 실행 로직
sequences = [os.path.join(RAW_DATA_DIR, d) for d in os.listdir(RAW_DATA_DIR)
             if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]

train_seqs, val_seqs = train_test_split(sequences, test_size=VAL_SIZE, random_state=42)

print(f"학습 시퀀스: {len(train_seqs)}개, 검증 시퀀스: {len(val_seqs)}개")

for seq in train_seqs: process_sequence(seq, 'train')
for seq in val_seqs: process_sequence(seq, 'val')

print("✅ 시퀀스별 데이터 통합 및 분할 완료!")