import json
import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# ==========================================================
# 1. 경로 설정 (보내주신 이미지 구조 반영)
# ==========================================================
BASE_PATH = r'C:\Users\WKU\Documents\Lee_Chung_Hyeon\OpenFieldAI-main\dataset' # 실제 경로로 수정하세요
BBOX_LABEL_DIR = os.path.join(BASE_PATH, '1.Training/labeling_data/TL_Bbox')
POLY_LABEL_DIR = os.path.join(BASE_PATH, '1.Training/labeling_data/TL_Polygon')
# 이미지 경로는 source_data 폴더명을 확인하여 수정하세요 (예: 1.Training/source_data)
IMAGE_ROOT_DIR = os.path.join(BASE_PATH, '1.Training/source_data')

OUTPUT_ROOT = os.path.join(BASE_PATH, 'hybridnets_final_dataset')
IMG_W, IMG_H = 1920, 1080

# 클래스 매핑 (진단 결과의 category_id 반영)
DET_CLASS_MAP = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}

for sub in ['images', 'labels', 'segmentation']:
    os.makedirs(os.path.join(OUTPUT_ROOT, sub), exist_ok=True)

def main():
    # 1. 모든 이미지 위치 미리 인덱싱 (하위 폴더가 많으므로 필수)
    print("🔍 원본 이미지 위치를 찾는 중입니다 (시간이 다소 소요될 수 있습니다)...")
    image_pool = {p.name: p for p in Path(IMAGE_ROOT_DIR).rglob('*') if p.suffix.lower() in ['.jpg', '.png']}
    print(f"✅ 총 {len(image_pool)}개의 이미지를 인덱싱했습니다.")

    # 2. 모든 _total.json 파일 찾기
    json_files = list(Path(BBOX_LABEL_DIR).rglob('*_total.json'))
    print(f"📂 처리할 JSON 파일: {len(json_files)}개")

    total_success = 0

    for json_path in json_files:
        print(f"\n📄 {json_path.name} 처리 중...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 이미지 ID와 파일명을 매핑
        id_to_file = {img['id']: img['file_name'] for img in data.get('images', [])}

        # 이미지 ID별로 어노테이션 그룹화
        annotations_by_id = {}
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in annotations_by_id:
                annotations_by_id[img_id] = []
            annotations_by_id[img_id].append(ann)

        # 각 이미지별로 처리 시작
        for img_id, file_name in tqdm(id_to_file.items(), desc="JSON 내 이미지 분리 중"):
            # A. 이미지 파일 복사
            if file_name in image_pool:
                src_path = image_pool[file_name]
                shutil.copy(src_path, os.path.join(OUTPUT_ROOT, 'images', file_name))
            else:
                continue # 이미지가 없으면 건너뜀

            # B. YOLO 라벨 생성
            yolo_lines = []
            if img_id in annotations_by_id:
                for ann in annotations_by_id[img_id]:
                    cat_id = ann['category_id']
                    if cat_id in DET_CLASS_MAP:
                        x, y, w, h = ann['bbox']
                        cx, cy, nw, nh = (x + w/2)/IMG_W, (y + h/2)/IMG_H, w/IMG_W, h/IMG_H
                        yolo_lines.append(f"{DET_CLASS_MAP[cat_id]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            # 라벨 파일 저장
            base_name = Path(file_name).stem
            with open(os.path.join(OUTPUT_ROOT, 'labels', f"{base_name}.txt"), 'w') as lf:
                lf.write("\n".join(yolo_lines))

            # C. Segmentation 마스크 생성 (기존 로직 유지하되 파일명 매칭 방식만 수정)
            # (이 단계에서는 Polygon용 JSON도 이와 같은 방식으로 열어서 처리해야 합니다)

            if yolo_lines: total_success += 1

    print(f"\n✨ 완료! 총 {total_success}세트의 데이터가 성공적으로 생성되었습니다.")

if __name__ == "__main__":
    main()