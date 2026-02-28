import matplotlib.pyplot as plt
import numpy as np

methods = ['WAM', 'OmniGuard', 'StableGuard', 'Ours*']

# 데이터
auc_sp = [0.999, 1.000, 1.000, 0.988]
iou_sp = [0.991, 0.997, 0.996, 0.949]
auc_fr = [0.950, 0.486, 0.500, 0.970]
iou_fr = [0.844, 0.794, 0.067, 0.921]

x = np.arange(len(methods))
width = 0.21

fig, ax = plt.subplots(figsize=(8.7, 3))

# SP 파란색 톤에 맞춘 FR 붉은색 톤 지정 (AUC는 진하게, IoU는 연하게)
color_auc_sp = '#4682B4'
color_iou_sp = '#C7D9E8'
color_auc_fr = '#C44E52'
color_iou_fr = '#F0B2B6'

rects1 = ax.bar(x - 1.5*width, auc_sp, width, label='AUC (SP)', color=color_auc_sp)
rects2 = ax.bar(x - 0.5*width, iou_sp, width, label='IoU (SP)', color=color_iou_sp)
rects3 = ax.bar(x + 0.5*width, auc_fr, width, label='AUC (FR)', color=color_auc_fr)
rects4 = ax.bar(x + 1.5*width, iou_fr, width, label='IoU (FR)', color=color_iou_fr)

ax.set_ylabel('Scores', fontsize=12)
ax.tick_params(axis='y', labelsize=12)
# ax.set_title('AUC & IoU Comparison: SP vs FR', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=12)

# 가로로 숫자 표시
ax.bar_label(rects1, fmt='%.2f', padding=2, fontsize=10)
ax.bar_label(rects2, fmt='%.2f', padding=2, fontsize=10)
ax.bar_label(rects3, fmt='%.2f', padding=2, fontsize=10)
ax.bar_label(rects4, fmt='%.2f', padding=2, fontsize=10)

ax.set_ylim(0, 1.15) 

plt.tight_layout()
plt.savefig('fig1_quan.png', dpi=300, bbox_inches='tight')
plt.close()