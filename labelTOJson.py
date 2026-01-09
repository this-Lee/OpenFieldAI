import json
import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# ==========================================================
# 1. 경로 및 설정 (사용자 환경에 맞게 수정하세요)
# ==========================================================
BASE_PATH = './dataset/training'
BBOX_LABEL_DIR = os.path.join(BASE_PATH, 'labeling_data/TL_Bbox')
POLY_LABEL_DIR = os.path.join(BASE_PATH, 'labeling_data/TL_Polygon')
IMAGE_DIR = os.path.join(BASE_PATH, 'source_data/TS_Bbox')

OUTPUT_ROOT = './datasets/hybridnets_data'
IMG_W, IMG_H = 1920, 1080

# 클래스 매핑 (ID -> 모델 학습 인덱스)
# 3:person, 4:vehicle, 5:rocks, 6:vail, 7:tractor, 8:pole, 9:tree
DET_CLASS_MAP = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}

# 폴더 생성
for sub in ['images', 'labels', 'segmentation']:
    os.makedirs(os.path.join(OUTPUT_ROOT, sub), exist_ok=True)

def convert_bbox(bbox, w, h):
    """[x, y, width, height] -> YOLO [cx, cy, w, h] 정규화"""
    x, y, bw, bh = bbox
    cx = (x + bw / 2.0) / w
    cy = (y + bh / 2.0) / h
    nw = bw / w
    nh = bh / h
    return [cx, cy, nw, nh]

def main():
    # 2. 파일 스캔 (하위 폴더 포함)
    print("📂 데이터를 스캔 중입니다. 잠시만 기다려주세요...")

    # 각 파일의 Stem(확장자 제외 이름)을 키로 전체 경로 저장
    # NIA 데이터의 'TL_', 'TS_', '_Bbox', '_Polygon' 접미사를 제거하여 매칭용 키 생성
    def get_clean_key(name):
        return name.replace('TL_', '').replace('TS_', '').replace('_Bbox', '').replace('_Polygon', '')

    bbox_jsons = {get_clean_key(p.stem): p for p in Path(BBOX_LABEL_DIR).rglob('*.json')}
    poly_jsons = {get_clean_key(p.stem): p for p in Path(POLY_LABEL_DIR).rglob('*.json')}
    image_pool = {get_clean_key(p.stem): p for p in Path(IMAGE_DIR).rglob('*')
                  if p.suffix.lower() in ['.jpg', '.jpeg', '.png']}

    common_keys = set(image_pool.keys())
    print(f"발견된 이미지: {len(image_pool)}개")
    print(f"매칭된 Bbox JSON: {len(bbox_jsons)}개")
    print(f"매칭된 Polygon JSON: {len(poly_jsons)}개")

    # 3. 통합 처리 루프
    print("🚀 데이터 통합 변환을 시작합니다...")

    for key in tqdm(common_keys):
        img_path = image_pool[key]
        img_filename = img_path.name
        base_name = img_path.stem

        # --- A. 이미지 복사 ---
        shutil.copy(img_path, os.path.join(OUTPUT_ROOT, 'images', img_filename))

        # --- B. Bbox 처리 (Detection) ---
        yolo_labels = []
        if key in bbox_jsons:
            with open(bbox_jsons[key], 'r', encoding='utf-8') as f:
                bbox_data = json.load(f)

            # JSON 구조에 따라 'annotations' 또는 'objects' 탐색
            objs = bbox_data.get('annotations', []) if 'annotations' in bbox_data else bbox_data.get('objects', [])
            for obj in objs:
                cat_id = obj.get('category_id') or obj.get('label')
                if cat_id in DET_CLASS_MAP:
                    bbox = obj.get('bbox')
                    if bbox:
                        yolo_box = convert_bbox(bbox, IMG_W, IMG_H)
                        yolo_labels.append(f"{DET_CLASS_MAP[cat_id]} " + " ".join([f"{v:.6f}" for v in yolo_box]))

        with open(os.path.join(OUTPUT_ROOT, 'labels', f"{base_name}.txt"), f"w") as f:
            f.write("\n".join(yolo_labels))

        # --- C. Polygon 처리 (Segmentation Mask) ---
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        if key in poly_jsons:
            with open(poly_jsons[key], 'r', encoding='utf-8') as f:
                poly_data = json.load(f)

            objs = poly_data.get('objects', []) if 'objects' in poly_data else poly_data.get('annotations', [])
            for obj in objs:
                # 'common_road' 라벨을 주행 영역(1)으로 설정
                if obj.get('label') == 'common_road' and 'position' in obj:
                    # 폴리곤 좌표가 리스트의 리스트 형태일 수 있으므로 처리
                    pos = obj['position']
                    if isinstance(pos[0], list): # [[x1,y1,x2,y2...]] 형태
                        pts = np.array(pos[0]).reshape(-1, 2).astype(np.int32)
                    else: # [x1,y1,x2,y2...] 형태
                        pts = np.array(pos).reshape(-1, 2).astype(np.int32)

                    cv2.fillPoly(mask, [pts], 1)

        # 마스크 저장 (.png)
        cv2.imwrite(os.path.join(OUTPUT_ROOT, 'segmentation', f"{base_name}.png"), mask)

    print(f"\n✅ 모든 공정이 완료되었습니다!")
    print(f"결과물 위치: {os.path.abspath(OUTPUT_ROOT)}")

if __name__ == "__main__":
    main()