import cv2
import pywt
import numpy as np
import os

def save_wavelet_components(image_path, wavelet='haar'):
    # 이미지를 흑백으로 불러오기 (컬러 분리 시 각 채널별 적용 필요)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("이미지 경로를 확인하세요.")

    # 2D 이산 웨이블릿 변환 (DWT) 수행
    coeffs2 = pywt.dwt2(img, wavelet)
    LL, (LH, HL, HH) = coeffs2

    # 시각화 및 저장을 위해 0~255 범위로 정규화하는 함수
    def normalize(data):
        data = np.abs(data)
        norm_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8) * 255
        return norm_data.astype(np.uint8)

    # 원본 파일명과 확장자 분리
    base_name, ext = os.path.splitext(image_path)
    
    # 4개 성분 이미지 저장
    components = {'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH}
    for name, data in components.items():
        save_path = f"{name}.png"
        cv2.imwrite(save_path, normalize(data))
        print(f"Saved: {save_path}")

# 사용 예시
# save_wavelet_components("/mnt/nas5/suhyeon/datasets/valAGE-Set/0018.png")
save_wavelet_components("bluesky_white2.png")