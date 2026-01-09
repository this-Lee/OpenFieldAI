import json
import os
import cv2
import numpy as np
import shutil
from pathlib import Path

# 1. 경로 설정 (사용자 환경에 맞게 수정)
BASE_PATH = './dataset/training'
LABEL_DIR = os.path.join(BASE_PATH, 'labeling_data/TL_Bbox') # 또는 TL_Polygon이 포함된 상위 폴더
IMAGE_DIR = os.path.join(BASE_PATH, 'source_data/TS_Bbox')    # 실제 이미지 위치

OUTPUT_ROOT = './datasets/hybridnets_ready'
IMG_W, IMG_H = 1920, 1080

# 클래스 매핑 (ID 기반)
DET_CLASS_MAP = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}

def convert_bbox(bbox, w, h):
    x, y, bw, bh = bbox
    return [(x + bw/2)/w, (y + bh/2)/h, bw/w, bh/h]

# 폴더 생성
for sub in ['images', 'labels', 'segmentation']:
    os.makedirs(os.path.join(OUTPUT_ROOT, sub), exist_ok=True)

# 2. 변환 로직
def process_all_files():
    # 모든 JSON 파일 찾기 (하위 폴더 포함)
    json_files = list(Path(LABEL_DIR).rglob('*.json'))
    print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다.")

    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 이미지 정보와 라벨 정보 매칭
        # NIA 데이터셋은 보통 data['images'] 리스트 안에 정보가 들어있음
        images_info = data.get('images', [])

        # 만약 JSON 하나가 이미지 하나에 대응하는 구조라면 (개별 JSON)
        if not images_info and 'image_id' in data:
            images_info = [data] # 단일 객체를 리스트로 감쌈

        for img_info in images_info:
            file_name = img_info.get('file_name') or img_info.get('name')
            if not file_name: continue

            base_name = Path(file_name).stem

            # A. 이미지 파일 복사
            # source_data 폴더에서 해당 이미지를 찾음
            src_img_path = os.path.join(IMAGE_DIR, file_name)
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, os.path.join(OUTPUT_ROOT, 'images', file_name))
            else:
                print(f"경고: 이미지를 찾을 수 없습니다 -> {src_img_path}")
                continue

            # B. Bbox 라벨링 처리 (YOLO 형식)
            yolo_labels = []
            # annotations가 별도로 있거나 objects 안에 있는 경우 대응
            annotations = data.get('annotations', []) if 'annotations' in data else img_info.get('objects', [])

            for ann in annotations:
                cat_id = ann.get('category_id') or ann.get('label')
                # 텍스트 라벨일 경우 ID로 변환하는 로직 추가 가능
                if cat_id in DET_CLASS_MAP:
                    bbox = ann.get('bbox')
                    if bbox:
                        yolo_box = convert_bbox(bbox, IMG_W, IMG_H)
                        yolo_labels.append(f"{DET_CLASS_MAP[cat_id]} " + " ".join([f"{v:.6f}" for v in yolo_box]))

            with open(os.path.join(OUTPUT_ROOT, 'labels', f"{base_name}.txt"), 'w') as f:
                f.write("\n".join(yolo_labels))

            # C. Segmentation 마스크 생성 (Polygon 데이터가 있다면)
            mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
            for obj in img_info.get('objects', []):
                if obj.get('label') == 'common_road':
                    poly = np.array(obj['position']).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 1)

            cv2.imwrite(os.path.join(OUTPUT_ROOT, 'segmentation', f"{base_name}.png"), mask)

    print(f"✅ 모든 데이터가 {OUTPUT_ROOT}에 통합되었습니다.")

if __name__ == "__main__":
    process_all_files()