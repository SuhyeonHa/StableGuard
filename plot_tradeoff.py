import matplotlib.pyplot as plt

# --- 데이터 설정 ---
# 7. Trade-off (tau)
tau_x = [0.05, 0.1, 0.15, 0.2]
tau_psnr = [33.44, 32.85, 32.42, 32.30]
tau_sp_iou = [0.95, 0.95, 0.94, 0.94]
tau_fr_iou = [0.82, 0.90, 0.91, 0.91]

# 8. eps
eps_x = ["12/255", "16/255", "20/255"]
eps_psnr = [33.20, 32.85, 32.58]
eps_sp_iou = [0.94, 0.95, 0.94]
eps_fr_iou = [0.89, 0.90, 0.91]

# 9. steps
steps_x = [50, 100, 150, 200]
steps_psnr = [32.53, 32.76, 32.80, 32.85]
steps_sp_iou = [0.95, 0.95, 0.97, 0.95]
steps_fr_iou = [0.91, 0.91, 0.92, 0.90]
steps_time = [14.34, 28.82, 42.52, 57.54]

# 10. loss
loss_x = [0.025, 0.05, 0.1, 0.2]
loss_psnr = [29.03, 30.73, 32.80, 34.76]
loss_sp_iou = [0.91, 0.93, 0.94, 0.86]
loss_fr_iou = [0.91, 0.92, 0.90, 0.63]

# --- 시각화 ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

def plot_performance(ax, x, psnr, sp_iou, fr_iou, title, xlabel, time=None):
    # 1. PSNR (왼쪽 축) - 점선으로 처리하여 배경 느낌 부여
    ax_psnr = ax
    ln1 = ax_psnr.plot(x, psnr, color='#d62728', marker='o', linestyle=':', label='PSNR')
    ax_psnr.set_ylabel('PSNR', color='#d62728', fontsize=11, fontweight='bold')
    
    # 2. IoU (오른쪽 축) - 실선으로 강조
    ax_iou = ax.twinx()
    ln2 = ax_iou.plot(x, sp_iou, color='#1f77b4', marker='s', linewidth=2, label='SP IoU')
    ln3 = ax_iou.plot(x, fr_iou, color='#2ca02c', marker='^', linewidth=2, label='FR IoU')
    ax_iou.set_ylim(0.6, 1.0)
    if ax.get_subplotspec().is_last_col(): # 오른쪽 끝 그래프만 Y라벨 표시
        ax_iou.set_ylabel('IoU Score', fontsize=11, fontweight='bold')

    # 3. Time (Steps 그래프 전용) - 막대로 배경에 배치
    lns = ln1 + ln2 + ln3
    if time is not None:
        ax_time = ax.twinx()
        ax_time.spines["right"].set_position(("axes", 1.15)) # 축 위치 조정
        bars = ax_time.bar(x, time, color='gray', alpha=0.2, width=10, label='Time (s)')
        ax_time.set_ylabel('Time (s)', color='gray')

    ax.set_title(title, fontsize=13, pad=15)
    ax.grid(True, axis='y', alpha=0.3)

# Subplot 그리기
plot_performance(axes[0], tau_x, tau_psnr, tau_sp_iou, tau_fr_iou, 'Trade-off (τ)', 'τ')
plot_performance(axes[1], eps_x, eps_psnr, eps_sp_iou, eps_fr_iou, 'Epsilon (eps)', 'eps')
plot_performance(axes[2], steps_x, steps_psnr, steps_sp_iou, steps_fr_iou, 'Steps', 'steps', time=steps_time)
plot_performance(axes[3], loss_x, loss_psnr, loss_sp_iou, loss_fr_iou, 'Loss Weight', 'loss')

plt.tight_layout()
plt.savefig('fig_tradeoff.png', dpi=300)
# plt.show()