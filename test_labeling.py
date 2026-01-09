import json
import os
from pathlib import Path

# 경로 설정 (실제 경로와 맞는지 확인!)
BASE_PATH = './dataset/training'
BBOX_LABEL_DIR = os.path.join(BASE_PATH, 'labeling_data/TL_Bbox')
IMAGE_DIR = os.path.join(BASE_PATH, 'source_data/TS_Bbox')

def diagnostic():
    # 1. 파일 이름 샘플 확인
    img_files = list(Path(IMAGE_DIR).rglob('*.jpg'))[:3]
    json_files = list(Path(BBOX_LABEL_DIR).rglob('*.json'))[:3]

    print("=== [1. 파일명 매칭 테스트] ===")
    print(f"이미지 샘플: {[f.name for f in img_files]}")
    print(f"JSON 샘플: {[f.name for f in json_files]}")

    if not json_files:
        print("❌ 에러: JSON 파일을 하나도 찾지 못했습니다. 경로를 확인하세요.")
        return

    # 2. JSON 내부 구조 정밀 분석
    print("\n=== [2. JSON 내부 구조 분석 (첫 번째 파일)] ===")
    with open(json_files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"최상위 키 목록: {list(data.keys())}")

    # 재귀적으로 'objects'나 'bbox' 단어 찾기
    def find_key_recursive(d, target, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                if target in k.lower():
                    print(f"✅ 발견된 키: '{path}{k}'")
                find_key_recursive(v, target, path + k + " > ")
        elif isinstance(d, list) and len(d) > 0:
            find_key_recursive(d[0], target, path + "[list] > ")

    find_key_recursive(data, "obj")
    find_key_recursive(data, "bbox")
    find_key_recursive(data, "cat")

    print("\n=== [3. 첫 번째 객체 샘플 데이터] ===")
    # 객체 데이터가 어디 있는지 수동으로 확인하기 위한 샘플 출력
    # (내용이 너무 길면 앞부분만 출력)
    print(json.dumps(data, indent=2, ensure_ascii=False)[:1000])

diagnostic()