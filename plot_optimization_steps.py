import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

def parse_training_log(file_path):
    """
    training_log.txt를 파싱하여 Step별 Similarity 데이터를 추출합니다.
    """
    # 데이터 저장소: steps[step_number] = {'base': [], 'clean': [], 'noisy': []}
    steps_data = defaultdict(lambda: {'base': [], 'clean': [], 'noisy': []})
    
    # 정규표현식 컴파일
    regex_base = re.compile(r"Base Cosine Similarity.*Mean:\s*([0-9.\-e]+)")
    regex_clean = re.compile(r"Cosine Similarity -.*Mean:\s*([0-9.\-e]+)")
    regex_noisy = re.compile(r"Cosine Similarity \(Noisy\).*Mean:\s*([0-9.\-e]+)")
    regex_step = re.compile(r"Step\s*(\d+),.*Loss:")

    current_metrics = {}
    
    print(f"Reading {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # 1. Base Cosine Similarity 파싱
                match_base = regex_base.search(line)
                if match_base:
                    current_metrics['base'] = float(match_base.group(1))
                    continue

                # 2. Cosine Similarity (Clean) 파싱
                match_clean = regex_clean.search(line)
                if match_clean:
                    current_metrics['clean'] = float(match_clean.group(1))
                    continue

                # 3. Cosine Similarity (Noisy) 파싱
                match_noisy = regex_noisy.search(line)
                if match_noisy:
                    current_metrics['noisy'] = float(match_noisy.group(1))
                    continue

                # 4. Step 파싱 및 데이터 저장
                match_step = regex_step.search(line)
                if match_step:
                    step = int(match_step.group(1))
                    
                    if 'base' in current_metrics and 'clean' in current_metrics and 'noisy' in current_metrics:
                        steps_data[step]['base'].append(current_metrics['base'])
                        steps_data[step]['clean'].append(current_metrics['clean'])
                        steps_data[step]['noisy'].append(current_metrics['noisy'])
                    
                    current_metrics = {}

    except FileNotFoundError:
        print(f"Error: 파일 '{file_path}'을 찾을 수 없습니다.")
        return None

    return steps_data

def convert_to_dataframe(steps_data):
    """
    파싱된 딕셔너리 데이터를 DataFrame으로 변환합니다.
    """
    records = []
    for step, metrics in steps_data.items():
        n_images = len(metrics['base'])
        for i in range(n_images):
            records.append({
                'Step': step,
                'Type': 'Clean',      # 이름을 짧게 수정 (출력 가독성 위함)
                'Value': metrics['base'][i]
            })
            records.append({
                'Step': step,
                'Type': 'Watermarked',     # 이름을 짧게 수정
                'Value': metrics['clean'][i]
            })
            # records.append({
            #     'Step': step,
            #     'Type': 'Noisy',     # 이름을 짧게 수정
            #     'Value': metrics['noisy'][i]
            # })
    
    return pd.DataFrame(records)

def plot_metrics(df, save_path='training_distribution.png'):
    """
    Line Plot을 그립니다.
    """
    plt.figure(figsize=(7, 7))
    sns.set_style("whitegrid")

    sns.lineplot(
        data=df, 
        x='Step', 
        y='Value', 
        hue='Type', 
        style='Type',
        markers=True, 
        dashes=False,
        linewidth=2.5
    )

    plt.title("Cosine Similarity during Optimization", fontsize=16, fontweight='bold')
    plt.ylabel("Cosine Similarity", fontsize=14)
    plt.xlabel("Optimization Step", fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved to {save_path}")

if __name__ == "__main__":
    # 1. 로그 파일 경로 설정
    log_file = "/mnt/nas5/suhyeon/projects/eval_spliceless/ours_full/20260104-075451/training_log.txt"
    csv_save_path = "/mnt/nas5/suhyeon/projects/eval_spliceless/ours_full/20260104-075451/parsed_metrics.csv"
    stats_save_path = "/mnt/nas5/suhyeon/projects/eval_spliceless/ours_full/20260104-075451/metrics_summary.csv"

    # 2. 파싱
    data = parse_training_log(log_file)
    
    if data:
        # 3. 데이터프레임 변환
        df = convert_to_dataframe(data)
        
        # === [추가 기능 1] 전체 Raw 데이터 CSV로 저장 ===
        df.to_csv(csv_save_path, index=False)
        print(f"Parsed data saved to {csv_save_path}")

        # === [추가 기능 2] 평균 및 표준편차 계산 ===
        # Step별, Type별로 그룹화하여 mean과 std 계산
        summary_stats = df.groupby(['Step', 'Type'])['Value'].agg(['mean', 'std']).reset_index()
        
        # 통계 데이터도 CSV로 저장
        summary_stats.to_csv(stats_save_path, index=False)
        print(f"Summary statistics saved to {stats_save_path}")

        # === [추가 기능 3] 콘솔 출력 (마지막 Step 기준) ===
        last_step = summary_stats['Step'].max()
        print(f"\n=== Final Statistics (Step {last_step}) ===")
        
        final_stats = summary_stats[summary_stats['Step'] == last_step]
        print(final_stats.to_string(index=False))

        # 전체 요약 통계 출력 (너무 길면 상위 10개만)
        print("\n=== All Steps Summary (Head 10) ===")
        print(summary_stats.head(10).to_string(index=False))

        # 4. 그리기
        plot_metrics(df)