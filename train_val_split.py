import os
import shutil
import random
from tqdm import tqdm

# 경로 설정
DATASET_ROOT = r'C:\Users\WKU\Documents\Lee_Chung_Hyeon\OpenFieldAI-main\dataset\hybridnets_final'
TRAIN_DIR = os.path.join(DATASET_ROOT, 'train')
VAL_DIR = os.path.join(DATASET_ROOT, 'val')

def split_dataset(split_ratio=0.8):
    image_list = [f for f in os.listdir(os.path.join(DATASET_ROOT, 'images')) if f.endswith(('.jpg', '.png'))]
    random.seed(42)
    random.shuffle(image_list)

    split_idx = int(len(image_list) * split_ratio)
    train_files = image_list[:split_idx]
    val_files = image_list[split_idx:]

    for split, files in [('train', train_files), ('val', val_files)]:
        for f in tqdm(files, desc=f"Moving {split} files"):
            # 이미지, 라벨, 세그멘테이션 파일 경로
            base_name = os.path.splitext(f)[0]

            # 폴더 생성
            for sub in ['images', 'labels', 'segmentation']:
                os.makedirs(os.path.join(DATASET_ROOT, split, sub), exist_ok=True)

            # 이동 (Copy 대신 Move를 써야 용량 관리에 유리합니다)
            shutil.move(os.path.join(DATASET_ROOT, 'images', f), os.path.join(DATASET_ROOT, split, 'images', f))
            shutil.move(os.path.join(DATASET_ROOT, 'labels', base_name + '.txt'), os.path.join(DATASET_ROOT, split, 'labels', base_name + '.txt'))
            shutil.move(os.path.join(DATASET_ROOT, 'segmentation', base_name + '.png'), os.path.join(DATASET_ROOT, split, 'segmentation', base_name + '.png'))

if __name__ == "__main__":
    split_dataset()