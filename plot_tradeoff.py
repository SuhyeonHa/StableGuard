import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 데이터 설정 (이미지 기반)
tau_x = [0.1, 0.15, 0.2]
tau_psnr = [32.85, 32.42, 32.30]
tau_sp = [0.95, 0.94, 0.94]
tau_fr = [0.90, 0.91, 0.91]

# eps에서 8/255 제외
eps_x = ["12/255", "16/255", "20/255"]
eps_psnr = [33.20, 32.85, 32.58]
eps_sp = [0.94, 0.95, 0.94]
eps_fr = [0.89, 0.90, 0.91]

steps_x = [50, 100, 150, 200]
steps_psnr = [32.53, 32.76, 32.80, 32.85]
steps_sp = [0.95, 0.95, 0.97, 0.95]
steps_fr = [0.91, 0.91, 0.92, 0.90]
steps_time = [14.34, 28.82, 42.52, 57.54]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

def plot_performance(ax, x, psnr, sp, fr, title, xlabel, time=None, is_first=False, is_last=False):
    # PSNR 축 설정
    ax_psnr = ax
    ln1 = ax_psnr.plot(x, psnr, color='#d62728', marker='o', linestyle=':', label='PSNR')
    if is_first:
        ax_psnr.set_ylabel('PSNR', color='#d62728', fontsize=24, fontweight='bold')
    ax_psnr.set_ylim(32.0, 33.5)
    ax_psnr.tick_params(axis='both', labelsize=18)
    
    # IoU 축 설정
    ax_iou = ax.twinx()
    ln2 = ax_iou.plot(x, sp, color='#1f77b4', marker='s', linewidth=2, label='SP (IoU)')
    ln3 = ax_iou.plot(x, fr, color='#2ca02c', marker='^', linewidth=2, label='FR (IoU)')
    ax_iou.set_ylim(0.75, 1.0)
    ax_iou.tick_params(axis='y', labelsize=18) # IoU 눈금 크기 조절
    
    if is_last:
        ax_iou.set_ylabel('IoU Score', fontsize=24, fontweight='bold')

    # Time 정보 (막대 및 텍스트 유지, 축 삭제)
    if time is not None:
        ax_time = ax.twinx()
        ax_time.set_yticks([]) # y축 눈금 삭제
        ax_time.spines["right"].set_visible(False) # y축 선 삭제
        
        bars = ax_time.bar(x, time, color='gray', alpha=0.15, width=0.4)
        ax_time.set_ylim(0, max(time) * 1.2)
        
        for bar, val in zip(bars, time):
            # x축에 가깝게 배치하기 위해 낮은 y값(2) 설정
            ax_time.text(bar.get_x() + bar.get_width()/2., 2,
                        f'{val}s', ha='center', va='bottom', 
                        fontsize=16, color='dimgray', fontweight='semibold')

    # ax.set_title(title, fontsize=18, pad=10, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=24)
    ax.grid(True, axis='y', alpha=0.3)

    ax_psnr.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_iou.yaxis.set_major_locator(MaxNLocator(nbins=5))
    return ln1, ln2, ln3

# 그래프 그리기
lns1 = plot_performance(axes[0], [str(v) for v in tau_x], tau_psnr, tau_sp, tau_fr, r'Trade-off ($\tau$)', r'$\tau$', is_first=True)
lns2 = plot_performance(axes[1], eps_x, eps_psnr, eps_sp, eps_fr, r'Epsilon ($\epsilon_\delta$)', r'$\epsilon_\delta$')
lns3 = plot_performance(axes[2], [str(v) for v in steps_x], steps_psnr, steps_sp, steps_fr, 'Steps', 'Steps', time=steps_time, is_last=True)

# 통합 범례 (하단 배치)
handles = lns1[0] + lns1[1] + lns1[2]
labels = [h.get_label() for h in handles]
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.14), fontsize=24)

plt.tight_layout()
plt.savefig('fig_tradeoff.pdf', dpi=300, bbox_inches='tight')