import json
import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# ==========================================================
# 1. 경로 설정 (사용자 환경에 맞게 폴더명을 확인하세요)
# ==========================================================
BASE_PATH = './dataset/training'

# 라벨 폴더들
BBOX_LABEL_DIR = os.path.join(BASE_PATH, 'labeling_data/TL_Bbox')
POLY_LABEL_DIR = os.path.join(BASE_PATH, 'labeling_data/TL_Polygon')

# 이미지 폴더들 (두 곳 모두에서 이미지를 찾습니다)
IMAGE_DIRS = [
    os.path.join(BASE_PATH, 'source_data/TS_Bbox'),
    os.path.join(BASE_PATH, 'source_data/TS_Polygon')
]

OUTPUT_ROOT = './datasets/hybridnets_final'
IMG_W, IMG_H = 1920, 1080

# 클래스 매핑 (ID -> 학습 인덱스)
DET_CLASS_MAP = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}

# 폴더 생성
for sub in ['images', 'labels', 'segmentation']:
    os.makedirs(os.path.join(OUTPUT_ROOT, sub), exist_ok=True)

def get_clean_key(name):
    """파일명에서 접두사/접미사를 제거하여 순수 키 생성"""
    return name.replace('TL_', '').replace('TS_', '').replace('_Bbox', '').replace('_Polygon', '').split('.')[0]

def convert_bbox(bbox, w, h):
    x, y, bw, bh = bbox
    return [(x + bw/2)/w, (y + bh/2)/h, bw/w, bh/h]

def main():
    # 2. 모든 데이터 미리 스캔 (Index 구축)
    print("🔍 모든 폴더를 스캔하여 인덱스를 생성 중...")

    # 이미지 스캔 (여러 폴더 대응)
    image_pool = {}
    for d in IMAGE_DIRS:
        if os.path.exists(d):
            for p in Path(d).rglob('*'):
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_pool[get_clean_key(p.stem)] = p

    # JSON 스캔
    bbox_jsons = list(Path(BBOX_LABEL_DIR).rglob('*.json')) if os.path.exists(BBOX_LABEL_DIR) else []
    poly_jsons = list(Path(POLY_LABEL_DIR).rglob('*.json')) if os.path.exists(POLY_LABEL_DIR) else []

    # 데이터 저장용 딕셔너리
    final_bboxes = {key: [] for key in image_pool.keys()}
    final_polygons = {key: [] for key in image_pool.keys()}

    # 3. Bbox JSON 파싱 (다양한 형식 대응)
    print("📦 Bbox 라벨 해석 중...")
    for j_path in bbox_jsons:
        with open(j_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 형식 1: COCO 스타일 (annotations 리스트)
        if 'annotations' in data and 'images' in data:
            id_to_key = {img['id']: get_clean_key(img['file_name']) for img in data['images']}
            for ann in data['annotations']:
                key = id_to_key.get(ann['image_id'])
                if key in final_bboxes:
                    cat_id = ann.get('category_id')
                    if cat_id in DET_CLASS_MAP and 'bbox' in ann:
                        final_bboxes[key].append((DET_CLASS_MAP[cat_id], ann['bbox']))

        # 형식 2: NIA 이미지 중심 스타일 (images 내에 objects)
        elif 'images' in data:
            for img in data['images']:
                key = get_clean_key(img.get('name') or img.get('file_name'))
                if key in final_bboxes:
                    for obj in img.get('objects', []):
                        cat_id = obj.get('category_id') or obj.get('label')
                        if str(cat_id).isdigit() and int(cat_id) in DET_CLASS_MAP:
                            if 'bbox' in obj: final_bboxes[key].append((DET_CLASS_MAP[int(cat_id)], obj['bbox']))

    # 4. Polygon JSON 파싱
    print("🎨 Polygon 라벨 해석 중...")
    for j_path in poly_jsons:
        with open(j_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # NIA Polygon 형식 (images 리스트 순회)
        imgs_list = data.get('images', [])
        for img in imgs_list:
            key = get_clean_key(img.get('name') or img.get('file_name'))
            if key in final_polygons:
                for obj in img.get('objects', []):
                    if obj.get('label') == 'common_road' and 'position' in obj:
                        final_polygons[key].append(obj['position'])

    # 5. 최종 파일 생성
    print("💾 통합 데이터셋 저장 중...")
    for key, img_path in tqdm(image_pool.items()):
        # 이미지 복사
        shutil.copy(img_path, os.path.join(OUTPUT_ROOT, 'images', img_path.name))

        # YOLO txt 저장
        labels = [f"{c} {' '.join(map(str, convert_bbox(b, IMG_W, IMG_H)))}" for c, b in final_bboxes[key]]
        with open(os.path.join(OUTPUT_ROOT, 'labels', f"{img_path.stem}.txt"), 'w') as f:
            f.write("\n".join(labels))

        # Mask png 저장
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        for poly in final_polygons[key]:
            # 다중 리스트 구조 대응
            pts = np.array(poly[0] if isinstance(poly[0], list) else poly).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        cv2.imwrite(os.path.join(OUTPUT_ROOT, 'segmentation', f"{img_path.stem}.png"), mask)

    print(f"\n✅ 완료! 총 {len(image_pool)}세트의 데이터가 통합되었습니다.")

if __name__ == "__main__":
    main()