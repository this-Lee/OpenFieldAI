import json
import os
import cv2
import numpy as np
import shutil
from pathlib import Path

# 1. 경로 설정
BASE_PATH = './dataset/training'
LABEL_DIR = os.path.join(BASE_PATH, 'labeling_data/TL_Bbox')
IMAGE_DIR = os.path.join(BASE_PATH, 'source_data/TS_Bbox')

OUTPUT_ROOT = './datasets/hybridnets_ready'
IMG_W, IMG_H = 1920, 1080

# 클래스 매핑
DET_CLASS_MAP = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}

def convert_bbox(bbox, w, h):
    x, y, bw, bh = bbox
    return [(x + bw/2)/w, (y + bh/2)/h, bw/w, bh/h]

# 폴더 생성
for sub in ['images', 'labels', 'segmentation']:
    os.makedirs(os.path.join(OUTPUT_ROOT, sub), exist_ok=True)

def process_nested_folders():
    # A. 이미지 인덱싱 (모든 하위 폴더의 이미지를 미리 스캔)
    print("모든 하위 폴더에서 이미지 파일을 검색 중입니다...")
    # 이미지 파일명(확장자 제외)을 키로, 전체 경로를 값으로 저장
    image_pool = {p.stem: p for p in Path(IMAGE_DIR).rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']}
    print(f"총 {len(image_pool)}개의 이미지 파일을 찾았습니다.")

    # B. 모든 하위 폴더에서 JSON 파일 검색
    json_files = list(Path(LABEL_DIR).rglob('*.json'))
    print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다.")

    count = 0
    for json_path in json_files:
        # 매칭용 이름 생성 (NIA 데이터 특성 반영: TL_을 TS_로 변경 시도)
        target_stem = json_path.stem.replace('TL_', 'TS_')

        # 만약 이미지 풀에 해당 이름이 있다면
        if target_stem in image_pool:
            src_img_path = image_pool[target_stem]
            img_file_name = src_img_path.name
        elif json_path.stem in image_pool: # 이름이 똑같은 경우
            src_img_path = image_pool[json_path.stem]
            img_file_name = src_img_path.name
        else:
            print(f"⚠️ 매칭 실패: {json_path.name}와 일치하는 이미지를 찾을 수 없음")
            continue

        # --- 여기서부터는 파일 처리 로직 ---
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. 이미지 복사
        shutil.copy(src_img_path, os.path.join(OUTPUT_ROOT, 'images', img_file_name))

        # 2. Bbox 처리
        yolo_labels = []
        # JSON 구조에 따라 'annotations' 또는 'objects' 탐색
        objs = data.get('annotations', []) if 'annotations' in data else data.get('objects', [])

        for ann in objs:
            cat_id = ann.get('category_id') or ann.get('label')
            if cat_id in DET_CLASS_MAP:
                bbox = ann.get('bbox')
                if bbox:
                    yolo_box = convert_bbox(bbox, IMG_W, IMG_H)
                    yolo_labels.append(f"{DET_CLASS_MAP[cat_id]} " + " ".join([f"{v:.6f}" for v in yolo_box]))

        with open(os.path.join(OUTPUT_ROOT, 'labels', f"{json_path.stem}.txt"), 'w') as f:
            f.write("\n".join(yolo_labels))

        # 3. Segmentation 처리
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        has_seg = False
        for obj in data.get('objects', []):
            if obj.get('label') == 'common_road' and 'position' in obj:
                poly = np.array(obj['position']).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)
                has_seg = True

        if has_seg:
            cv2.imwrite(os.path.join(OUTPUT_ROOT, 'segmentation', f"{json_path.stem}.png"), mask)

        count += 1
        if count % 100 == 0:
            print(f"현재 {count}번째 파일 처리 중...")

    print(f"✅ 모든 처리가 완료되었습니다! (성공: {count}건)")

if __name__ == "__main__":
    process_nested_folders()