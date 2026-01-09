import json
import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# 1. 경로 설정
BASE_PATH = './dataset/training'
BBOX_LABEL_DIR = os.path.join(BASE_PATH, 'labeling_data/TL_Bbox')
POLY_LABEL_DIR = os.path.join(BASE_PATH, 'labeling_data/TL_Polygon')
IMAGE_DIRS = [
    os.path.join(BASE_PATH, 'source_data/TS_Bbox'),
    os.path.join(BASE_PATH, 'source_data/TS_Polygon')
]

OUTPUT_ROOT = './datasets/hybridnets_final'
IMG_W, IMG_H = 1920, 1080

# [중요] 클래스 매핑 테이블 (ID와 이름을 모두 등록)
DET_CLASS_MAP = {
    # 숫자 ID 대응
    3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6,
    '3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, '03': 0, '08': 5,
    # 텍스트 이름 대응
    'common_person': 0, 'common_vehicle': 1, 'common_rocks': 2,
    'common_vail': 3, 'common_tractor': 4, 'common_pole': 5, 'common_tree': 6,
    'person': 0, 'vehicle': 1, 'rocks': 2, 'vail': 3, 'tractor': 4, 'pole': 5, 'tree': 6
}

os.makedirs(os.path.join(OUTPUT_ROOT, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, 'labels'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, 'segmentation'), exist_ok=True)

def get_clean_key(name):
    name = name.split('.')[0]
    for p in ['TL_', 'TS_']: name = name.replace(p, '')
    for s in ['_Bbox', '_Polygon', '_RGB_middle', '_RGB_bottom']: name = name.replace(s, '')
    return name

def extract_bbox(obj):
    """다양한 형태의 Bbox 좌표를 [x, y, w, h]로 통일"""
    b = obj.get('bbox') or obj.get('coordinate') or obj.get('box2d')
    if isinstance(b, list) and len(b) == 4: return b
    if isinstance(b, dict): return [b.get('x', 0), b.get('y', 0), b.get('w') or b.get('width', 0), b.get('h') or b.get('height', 0)]
    return None

def main():
    print("🔍 데이터 스캔 중...")
    image_pool = {}
    for d in IMAGE_DIRS:
        if os.path.exists(d):
            for p in Path(d).rglob('*'):
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_pool[get_clean_key(p.stem)] = p

    bbox_jsons = {get_clean_key(p.stem): p for p in Path(BBOX_LABEL_DIR).rglob('*.json')}
    poly_jsons = {get_clean_key(p.stem): p for p in Path(POLY_LABEL_DIR).rglob('*.json')}

    success_count = 0
    for key, img_path in tqdm(image_pool.items(), desc="통합 작업"):
        shutil.copy(img_path, os.path.join(OUTPUT_ROOT, 'images', img_path.name))

        # --- 1. Bbox (Detection) 처리 ---
        yolo_labels = []
        if key in bbox_jsons:
            with open(bbox_jsons[key], 'r', encoding='utf-8') as f:
                data = json.load(f)

            # JSON 내부 객체 리스트 위치 자동 탐색
            objs = data.get('annotations') or data.get('objects') or \
                (data.get('learning_data_info', {}).get('objects', [])) or \
                (data.get('images', [{}])[0].get('objects', []))

            for obj in objs:
                # 라벨 식별 (ID 또는 이름)
                raw_id = obj.get('category_id') or obj.get('label')
                if raw_id in DET_CLASS_MAP:
                    coords = extract_bbox(obj)
                    if coords:
                        x, y, w, h = coords
                        cx, cy, nw, nh = (x + w/2)/IMG_W, (y + h/2)/IMG_H, w/IMG_W, h/IMG_H
                        yolo_labels.append(f"{DET_CLASS_MAP[raw_id]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(os.path.join(OUTPUT_ROOT, 'labels', f"{img_path.stem}.txt"), 'w') as f:
            f.write("\n".join(yolo_labels))

        # --- 2. Polygon (Segmentation) 처리 ---
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        if key in poly_jsons:
            with open(poly_jsons[key], 'r', encoding='utf-8') as f:
                p_data = json.load(f)
            p_objs = p_data.get('objects') or p_data.get('annotations') or \
                (p_data.get('images', [{}])[0].get('objects', []))

            for p_obj in p_objs:
                if 'road' in str(p_obj.get('label', '')).lower():
                    pos = p_obj.get('position')
                    if pos:
                        pts = np.array(pos).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [pts], 1)

        cv2.imwrite(os.path.join(OUTPUT_ROOT, 'segmentation', f"{img_path.stem}.png"), mask)
        if yolo_labels: success_count += 1

    print(f"\n✅ 완료! 라벨이 정상 추출된 이미지: {success_count} / {len(image_pool)}")

if __name__ == "__main__":
    main()