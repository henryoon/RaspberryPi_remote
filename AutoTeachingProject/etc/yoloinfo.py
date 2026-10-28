import torch
import os
import datetime

model_path = '/home/rnd/yolo_model/best_yolov26s.pt'
ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
print(ckpt.keys()) # 체크포인트 내 저장된 키 목록 확인

if 'train_args' in ckpt:
    print(f"Base Model: {ckpt['train_args'].get('model')}") # 학습 당시의 설정 (어떤 모델 기반인지, epoch 등)
    # print(f"Data/Args: {ckpt['train_args']}")
    
file_time = os.path.getmtime(model_path)
print(f"Model Last Modified: {datetime.datetime.fromtimestamp(file_time)}")

if 'license' in ckpt:
    print(f"License Info: {ckpt['license']}")
    