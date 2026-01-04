import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. 데이터 정의 (제공해주신 통계치 입력)
data = {
    'Layer': ['Layer 0', 'Layer 1', 'Layer 2', 'Layer 3'] * 3,
    'Method': ['1. Gaussian'] * 4 + ['2. Zero-Mean'] * 4 + ['3. Sign (Ours)'] * 4,
    'Mean': [
        -0.000363, 0.008614, 0.008382, 0.003605,  # Gaussian
        0.000501, 0.008518, 0.008221, 0.003647,   # Zero-Mean
        -0.003867, 0.007376, 0.009670, 0.003264   # Sign (Ours)
    ],
    'Std': [
        0.092337, 0.071615, 0.053094, 0.036312,   # Gaussian
        0.092288, 0.071611, 0.053249, 0.036324,   # Zero-Mean
        0.089113, 0.071766, 0.053242, 0.036300    # Sign (Ours)
    ]
}

df = pd.DataFrame(data)

# 2. 그래프 설정
fig, ax = plt.subplots(figsize=(10, 6))

# X축 위치 설정 (그룹화)
layers = df['Layer'].unique()
x = np.arange(len(layers))  # [0, 1, 2, 3]
width = 0.25  # 막대 간격

# 3. 플롯 그리기 (Error Bar)
# 각 Method별로 위치를 조금씩 이동(jitter)시켜서 그립니다.
methods = ['1. Gaussian', '2. Zero-Mean', '3. Sign (Ours)']
colors = ['#888888', '#4477AA', '#CC3333']  # 회색, 파랑, 빨강(강조)
markers = ['o', 's', 'D'] # 원, 사각형, 다이아몬드

for i, method in enumerate(methods):
    subset = df[df['Method'] == method]
    offset = (i - 1) * width  # -0.25, 0, +0.25
    
    ax.errorbar(
        x + offset, 
        subset['Mean'], 
        yerr=subset['Std'], 
        fmt=markers[i],       # 마커 모양
        capsize=5,            # 에러바 끝 가로선 길이
        capthick=1.5,
        elinewidth=1.5,
        color=colors[i],
        label=method,
        markersize=8
    )

# 4. 스타일링
ax.set_title("Orthogonality & Stability across Layers", fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel("Cosine Similarity (Mean ± Std)", fontsize=12)
ax.set_xlabel("Network Depth (Feature Dimension)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(Std $\\approx$ {df[df['Layer']==l]['Std'].iloc[0]:.3f})" for l in layers])

# 기준선 (0 = 직교)
ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Perfect Orthogonality')

# 그리드 및 범례
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.legend(title="Method", fontsize=10, loc='upper right')

# Y축 범위 (분포가 잘 보이도록)
ax.set_ylim(-0.15, 0.15)

plt.tight_layout()
plt.savefig("./orthogonality_experiment.png", dpi=300)