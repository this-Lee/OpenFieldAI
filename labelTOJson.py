import json
import os
import cv2
import numpy as np
from pathlib import Path

# 1. 설정 (사용자 경로에 맞게 수정하세요)
BBOX_JSON = 'bbox_data.json'      # Bbox & Categories 정보가 있는 JSON
POLY_JSON = 'polygon_data.json'   # Polygon 정보가 있는 JSON
OUTPUT_DIR = './datasets/field_data'
IMG_WIDTH, IMG_HEIGHT = 1920, 1080

# 클래스 매핑 (ID를 0부터 시작하도록 재배열)
# 3: person, 4: vehicle, 5: rocks, 6: vail, 7: tractor, 8: pole, 9: tree
DET_CLASS_MAP = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}

# 폴더 생성
for sub in ['images', 'labels', 'segmentation']:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

def convert_to_yolo(bbox, w, h):
    """COCO [x, y, width, height] -> YOLO [cx, cy, w, h] 정규화"""
    x, y, bw, bh = bbox
    cx = (x + bw / 2.0) / w
    cy = (y + bh / 2.0) / h
    nw = bw / w
    nh = bh / h
    return cx, cy, nw, nh

def process_data():
    # 데이터 로드
    with open(BBOX_JSON, 'r', encoding='utf-8') as f:
        bbox_data = json.load(f)
    with open(POLY_JSON, 'r', encoding='utf-8') as f:
        poly_data = json.load(f)

    # 1. Bbox & Image 메타데이터 처리
    img_id_map = {img['id']: img['file_name'] for img in bbox_data['images']}

    # 이미지별 라벨 저장용 딕셔너리
    labels_dict = {img_id: [] for img_id in img_id_map.keys()}

    for ann in bbox_data['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        if cat_id in DET_CLASS_MAP:
            yolo_bbox = convert_to_yolo(ann['bbox'], IMG_WIDTH, IMG_HEIGHT)
            labels_dict[img_id].append(f"{DET_CLASS_MAP[cat_id]} {' '.join(map(str, yolo_bbox))}")

    # 2. Polygon 데이터 처리 (Segmentation Mask 생성)
    # Polygon JSON의 images 리스트를 순회
    for p_img in poly_data['images']:
        # 파일명이 Bbox 데이터와 일치하는지 확인 (이름 기준 매칭)
        file_name = p_img['name']
        base_name = Path(file_name).stem

        # 빈 마스크 생성 (0: 배경)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

        for obj in p_img.get('objects', []):
            if obj['label'] == 'common_road':
                # 폴리곤 좌표 변환 및 채우기 (1: 주행 가능 영역)
                poly = np.array(obj['position']).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)

        # 마스크 저장
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'segmentation', f"{base_name}.png"), mask)

    # 3. 최종 YOLO 라벨 텍스트 저장
    for img_id, file_name in img_id_map.items():
        base_name = Path(file_name).stem
        with open(os.path.join(OUTPUT_DIR, 'labels', f"{base_name}.txt"), 'w') as f:
            f.write("\n".join(labels_dict[img_id]))

    print(f"✅ 변환 완료! 데이터셋이 {OUTPUT_DIR}에 저장되었습니다.")

if __name__ == "__main__":
    process_data()