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

# 클래스 매핑 (문자열과 숫자 모두 대응하도록 수정)
DET_CLASS_MAP = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}

for sub in ['images', 'labels', 'segmentation']:
    os.makedirs(os.path.join(OUTPUT_ROOT, sub), exist_ok=True)

def get_clean_key(name):
    # NIA 특유의 모든 접두사/접미사 제거 로직 강화
    name = name.split('.')[0]
    for prefix in ['TL_', 'TS_']: name = name.replace(prefix, '')
    for suffix in ['_Bbox', '_Polygon', '_RGB_middle', '_RGB_bottom']: name = name.replace(suffix, '')
    return name

def find_objects_in_json(data):
    """JSON 내부에서 객체 리스트를 어떻게든 찾아내는 함수"""
    # 1. NIA 표준: learning_data_info -> objects
    if isinstance(data, dict) and 'learning_data_info' in data:
        return data['learning_data_info'].get('objects', [])
    # 2. 이미지 리스트 형태: images -> [ { objects: [...] } ]
    if 'images' in data and isinstance(data['images'], list) and len(data['images']) > 0:
        # 첫 번째 이미지의 objects를 가져오거나 전체를 합침
        all_objs = []
        for img in data['images']:
            all_objs.extend(img.get('objects', []) or img.get('annotations', []))
        return all_objs
    # 3. 최상위 annotations 또는 objects
    return data.get('annotations', []) or data.get('objects', [])

def main():
    print("🔍 데이터 스캔 및 인덱싱 시작...")
    image_pool = {}
    for d in IMAGE_DIRS:
        if os.path.exists(d):
            for p in Path(d).rglob('*'):
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_pool[get_clean_key(p.stem)] = p

    bbox_jsons = {get_clean_key(p.stem): p for p in Path(BBOX_LABEL_DIR).rglob('*.json')}
    poly_jsons = {get_clean_key(p.stem): p for p in Path(POLY_LABEL_DIR).rglob('*.json')}

    print(f"이미지: {len(image_pool)}개 | Bbox JSON: {len(bbox_jsons)}개 | Polygon JSON: {len(poly_jsons)}개")

    success_count = 0
    for key, img_path in tqdm(image_pool.items(), desc="통합 변환 중"):
        # 1. 이미지 복사
        shutil.copy(img_path, os.path.join(OUTPUT_ROOT, 'images', img_path.name))

        # 2. Bbox 처리
        yolo_labels = []
        if key in bbox_jsons:
            with open(bbox_jsons[key], 'r', encoding='utf-8') as f:
                data = json.load(f)

            objs = find_objects_in_json(data)
            for obj in objs:
                # category_id 추출 (문자열/숫자 모두 대응)
                c_id = obj.get('category_id') or obj.get('label')
                try:
                    c_id = int(c_id)
                    if c_id in DET_CLASS_MAP:
                        # bbox 키 이름 대응 (bbox 또는 coordinate)
                        bbox = obj.get('bbox') or obj.get('coordinate')
                        if bbox:
                            x, y, w, h = bbox
                            cx, cy, nw, nh = (x + w/2)/IMG_W, (y + h/2)/IMG_H, w/IMG_W, h/IMG_H
                            yolo_labels.append(f"{DET_CLASS_MAP[c_id]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                except: continue

        with open(os.path.join(OUTPUT_ROOT, 'labels', f"{img_path.stem}.txt"), 'w') as f:
            f.write("\n".join(yolo_labels))

        # 3. Polygon 처리
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        if key in poly_jsons:
            with open(poly_jsons[key], 'r', encoding='utf-8') as f:
                data = json.load(f)
            objs = find_objects_in_json(data)
            for obj in objs:
                if 'common_road' in str(obj.get('label', '')) and 'position' in obj:
                    pos = obj['position']
                    # 좌표가 [x,y,x,y...] 인지 [[x,y],[x,y]...] 인지 자동 판별
                    pts = np.array(pos).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)

        cv2.imwrite(os.path.join(OUTPUT_ROOT, 'segmentation', f"{img_path.stem}.png"), mask)
        if len(yolo_labels) > 0: success_count += 1

    print(f"\n✅ 완료! 라벨이 생성된 이미지: {success_count}개 / 전체: {len(image_pool)}개")

if __name__ == "__main__":
    main()