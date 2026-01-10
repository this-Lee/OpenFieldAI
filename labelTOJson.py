import json
import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# 1. 경로 설정 (사용자님의 PC 환경 반영)
BASE_PATH = r'C:\Users\WKU\Documents\Lee_Chung_Hyeon\OpenFieldAI-main\dataset\1.Training'
BBOX_ROOT = os.path.join(BASE_PATH, 'labeling_data/TL_Bbox')
POLY_ROOT = os.path.join(BASE_PATH, 'labeling_data/TL_Polygon')
IMAGE_ROOT = os.path.join(BASE_PATH, 'source_data') # TS_Bbox, TS_Polygon 포함 상위 폴더

OUTPUT_ROOT = r'C:\Users\WKU\Documents\Lee_Chung_Hyeon\OpenFieldAI-main\dataset\hybridnets_final'
IMG_W, IMG_H = 1920, 1080

# 클래스 매핑
DET_CLASS_MAP = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}

os.makedirs(os.path.join(OUTPUT_ROOT, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, 'labels'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, 'segmentation'), exist_ok=True)

def main():
    # A. 이미지 위치 인덱싱 (30만 장 대응을 위해 파일명 중심 인덱싱)
    print("🔍 모든 원본 이미지 위치를 스캔 중입니다...")
    image_pool = {p.name: p for p in Path(IMAGE_ROOT).rglob('*') if p.suffix.lower() in ['.jpg', '.png']}
    print(f"✅ 총 {len(image_pool)}개의 이미지를 발견했습니다.")

    # B. 모든 Bbox 및 Polygon Total JSON 파일 찾기
    bbox_total_jsons = list(Path(BBOX_ROOT).rglob('*_total.json'))
    poly_total_jsons = list(Path(POLY_ROOT).rglob('*_total.json'))

    # C. Polygon 데이터를 파일명 기준으로 사전 로드 (메모리 최적화를 위해 시퀀스별 매칭 권장)
    # 여기서는 모든 Polygon 데이터를 파일명:어노테이션 구조로 임시 저장합니다.
    poly_lookup = {}
    print("🎨 Polygon 정보를 사전 분석 중입니다...")
    for p_json in tqdm(poly_total_jsons, desc="Polygon JSON 읽는 중"):
        with open(p_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            id_to_name = {img['id']: img['file_name'] for img in data.get('images', [])}
            for ann in data.get('annotations', []):
                fname = id_to_name.get(ann['image_id'])
                if fname:
                    if fname not in poly_lookup: poly_lookup[fname] = []
                    # segmentation 좌표 데이터 저장
                    if 'segmentation' in ann:
                        poly_lookup[fname].append(ann['segmentation'])

    # D. Bbox JSON을 기준으로 메인 루프 실행
    print("🚀 통합 변환 작업을 시작합니다...")
    success_count = 0

    for b_json in bbox_total_jsons:
        with open(b_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        id_to_name = {img['id']: img['file_name'] for img in data.get('images', [])}

        # Bbox 그룹화
        bbox_by_name = {}
        for ann in data.get('annotations', []):
            fname = id_to_name.get(ann['image_id'])
            if fname:
                if fname not in bbox_by_name: bbox_by_name[fname] = []
                bbox_by_name[fname].append(ann)

        for fname in tqdm(id_to_name.values(), desc=f"처리 중: {b_json.name}"):
            if fname not in image_pool: continue

            # 1. 이미지 복사
            shutil.copy(image_pool[fname], os.path.join(OUTPUT_ROOT, 'images', fname))
            base_name = Path(fname).stem

            # 2. Bbox 처리 (labels/*.txt)
            yolo_lines = []
            if fname in bbox_by_name:
                for ann in bbox_by_name[fname]:
                    c_id = ann.get('category_id')
                    if c_id in DET_CLASS_MAP:
                        x, y, w, h = ann['bbox']
                        cx, cy, nw, nh = (x + w/2)/IMG_W, (y + h/2)/IMG_H, w/IMG_W, h/IMG_H
                        yolo_lines.append(f"{DET_CLASS_MAP[c_id]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            with open(os.path.join(OUTPUT_ROOT, 'labels', f"{base_name}.txt"), 'w') as lf:
                lf.write("\n".join(yolo_lines))

            # 3. Polygon 처리 (segmentation/*.png)
            mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
            if fname in poly_lookup:
                for seg in poly_lookup[fname]:
                    # COCO 포맷은 [[x1,y1,x2,y2...]] 형태임
                    for poly in seg:
                        pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [pts], 1) # 도로 영역을 1로 채움

            cv2.imwrite(os.path.join(OUTPUT_ROOT, 'segmentation', f"{base_name}.png"), mask)
            success_count += 1

    print(f"✅ 최종 완료! 생성된 세트: {success_count}개")

if __name__ == "__main__":
    main()