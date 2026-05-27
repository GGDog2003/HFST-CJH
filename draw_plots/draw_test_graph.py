import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def parse_training_log(log_text):
    """
    解析训练日志，提取训练loss和验证集指标
    """
    # 提取训练迭代的loss
    train_pattern = r'<epoch:\s*(\d+),\s*iter:\s*([\d,]+),\s*lr:([\d.]+e[+-]\d+)>\s*G_loss:\s*([\d.]+e[+-]\d+)'
    train_matches = re.findall(train_pattern, log_text)

    iterations = []
    losses = []
    epochs = []

    for match in train_matches:
        epoch = int(match[0])
        iteration = int(match[1].replace(',', ''))
        lr = float(match[2])
        loss = float(match[3])

        epochs.append(epoch)
        iterations.append(iteration)
        losses.append(loss)

    # 提取验证集指标
    val_pattern = r'<epoch:\s*(\d+),\s*iter:\s*([\d,]+),\s*Average PSNR :\s*([\d.]+)dB\|([\d.]+)'
    val_matches = re.findall(val_pattern, log_text)

    val_iterations = []
    val_psnr = []
    val_ssim = []
    val_epochs = []

    for match in val_matches:
        epoch = int(match[0])
        iteration = int(match[1].replace(',', ''))
        psnr = float(match[2])
        ssim = float(match[3])

        val_epochs.append(epoch)
        val_iterations.append(iteration)
        val_psnr.append(psnr)
        val_ssim.append(ssim)

    return {
        'train': {
            'epochs': epochs,
            'iterations': iterations,
            'losses': losses
        },
        'val': {
            'epochs': val_epochs,
            'iterations': val_iterations,
            'psnr': val_psnr,
            'ssim': val_ssim
        }
    }

def smooth_curve(values, weight=0.8):
    """
    使用指数移动平均平滑曲线
    """
    smoothed = []
    last = values[0]
    for value in values:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_training_curves(log_text, save_path=None, smooth_weight=0.9):
    """
    绘制训练曲线：Loss图、PSNR-Epoch图、SSIM-Epoch图
    """
    data = parse_training_log(log_text)

    # 创建图表：1行3列
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')

    # 1. Loss曲线（横轴为iteration）
    ax1 = axes[0]
    ax1.plot(data['train']['iterations'], data['train']['losses'],
             alpha=0.3, color='blue', linewidth=0.5, label='Original Loss')
    smoothed_loss = smooth_curve(data['train']['losses'], weight=smooth_weight)
    ax1.plot(data['train']['iterations'], smoothed_loss,
             color='blue', linewidth=2, label='Smoothed Loss')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('G_Loss', fontsize=12)
    ax1.set_title('Generator Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. PSNR随Epoch变化图
    ax2 = axes[1]
    ax2.plot(data['val']['epochs'], data['val']['psnr'],
             marker='o', color='green', linewidth=2, markersize=8,
             markerfacecolor='white', markeredgewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('Average PSNR vs Epoch', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 在PSNR图上标注数值
    for i, (x, y) in enumerate(zip(data['val']['epochs'], data['val']['psnr'])):
        ax2.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))

    # 3. SSIM随Epoch变化图
    ax3 = axes[2]
    ax3.plot(data['val']['epochs'], data['val']['ssim'],
             marker='s', color='red', linewidth=2, markersize=8,
             markerfacecolor='white', markeredgewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('SSIM', fontsize=12)
    ax3.set_title('Average SSIM vs Epoch', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 在SSIM图上标注数值
    for i, (x, y) in enumerate(zip(data['val']['epochs'], data['val']['ssim'])):
        ax3.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

    return data

def plot_loss_and_validation(log_text, save_path=None, smooth_weight=0.9):
    """
    分别绘制三个独立的图：Loss、PSNR-Epoch、SSIM-Epoch
    """
    data = parse_training_log(log_text)

    # 图1：Loss曲线
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(data['train']['iterations'], data['train']['losses'],
             alpha=0.3, color='blue', linewidth=0.5, label='Original Loss')
    smoothed_loss = smooth_curve(data['train']['losses'], weight=smooth_weight)
    ax1.plot(data['train']['iterations'], smoothed_loss,
             color='blue', linewidth=2, label='Smoothed Loss')
    ax1.set_xlabel('Iteration', fontsize=14)
    ax1.set_ylabel('G_Loss', fontsize=14)
    ax1.set_title('Generator Loss During Training', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_loss.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 图2：PSNR随Epoch变化
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(data['val']['epochs'], data['val']['psnr'],
             marker='o', color='green', linewidth=2.5, markersize=10,
             markerfacecolor='lightgreen', markeredgewidth=2, markeredgecolor='darkgreen')
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('PSNR (dB)', fontsize=14)
    ax2.set_title('Average PSNR on Validation Set', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 标注最大值
    max_psnr_idx = data['val']['psnr'].index(max(data['val']['psnr']))
    ax2.annotate(f'Best: {data["val"]["psnr"][max_psnr_idx]:.2f} dB\n(Epoch {data["val"]["epochs"][max_psnr_idx]})',
                 xy=(data['val']['epochs'][max_psnr_idx], data['val']['psnr'][max_psnr_idx]),
                 xytext=(data['val']['epochs'][max_psnr_idx]+5, data['val']['psnr'][max_psnr_idx]-0.2),
                 arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                 fontsize=12, color='darkgreen', fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_psnr.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 图3：SSIM随Epoch变化
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(data['val']['epochs'], data['val']['ssim'],
             marker='s', color='red', linewidth=2.5, markersize=10,
             markerfacecolor='lightcoral', markeredgewidth=2, markeredgecolor='darkred')
    ax3.set_xlabel('Epoch', fontsize=14)
    ax3.set_ylabel('SSIM', fontsize=14)
    ax3.set_title('Average SSIM on Validation Set', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # 标注最大值
    max_ssim_idx = data['val']['ssim'].index(max(data['val']['ssim']))
    ax3.annotate(f'Best: {data["val"]["ssim"][max_ssim_idx]:.4f}\n(Epoch {data["val"]["epochs"][max_ssim_idx]})',
                 xy=(data['val']['epochs'][max_ssim_idx], data['val']['ssim'][max_ssim_idx]),
                 xytext=(data['val']['epochs'][max_ssim_idx]+5, data['val']['ssim'][max_ssim_idx]-0.005),
                 arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                 fontsize=12, color='darkred', fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_ssim.png'), dpi=300, bbox_inches='tight')
    plt.show()

    return data

def print_training_summary(data):
    """
    打印训练摘要信息
    """
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    # 训练信息
    print(f"\nTraining Statistics:")
    print(f"  Total iterations: {data['train']['iterations'][-1]}")
    print(f"  Total epochs: {data['train']['epochs'][-1]}")
    print(f"  Initial loss: {data['train']['losses'][0]:.6f}")
    print(f"  Final loss: {data['train']['losses'][-1]:.6f}")

    # 验证集信息
    print(f"\nValidation Statistics:")
    print(f"  Number of validation points: {len(data['val']['epochs'])}")

    max_psnr_idx = data['val']['psnr'].index(max(data['val']['psnr']))
    max_ssim_idx = data['val']['ssim'].index(max(data['val']['ssim']))

    print(f"  Best PSNR: {max(data['val']['psnr']):.2f} dB (Epoch {data['val']['epochs'][max_psnr_idx]})")
    print(f"  Best SSIM: {max(data['val']['ssim']):.4f} (Epoch {data['val']['epochs'][max_ssim_idx]})")
    print(f"  Final PSNR: {data['val']['psnr'][-1]:.2f} dB (Epoch {data['val']['epochs'][-1]})")
    print(f"  Final SSIM: {data['val']['ssim'][-1]:.4f} (Epoch {data['val']['epochs'][-1]})")

    # PSNR和SSIM提升
    psnr_improvement = data['val']['psnr'][-1] - data['val']['psnr'][0]
    ssim_improvement = data['val']['ssim'][-1] - data['val']['ssim'][0]
    print(f"\nImprovement (from first to last validation):")
    print(f"  PSNR improvement: {psnr_improvement:+.2f} dB")
    print(f"  SSIM improvement: {ssim_improvement:+.4f}")

    # 打印每个验证点的详细信息
    print(f"\nValidation Details:")
    print(f"{'Epoch':<8} {'PSNR(dB)':<12} {'SSIM':<10}")
    print("-" * 30)
    for i in range(len(data['val']['epochs'])):
        print(f"{data['val']['epochs'][i]:<8} {data['val']['psnr'][i]:<12.2f} {data['val']['ssim'][i]:<10.4f}")

# 主程序示例
if __name__ == "__main__":
    # 方式1：读取日志文件
    log_file_path = "tmp_training_log.txt"  # 替换为你的日志文件路径

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_text = f.read()
        print(f"Successfully loaded log file: {log_file_path}")
    except FileNotFoundError:
        # 方式2：直接粘贴日志文本
        print("Log file not found. Using hardcoded log text...")
        log_text = """
        PASTE YOUR TRAINING LOG HERE
        """

    # 绘制合并图表（Loss + PSNR-Epoch + SSIM-Epoch）
    data = plot_training_curves(log_text, save_path='training_curves_combined.png')

    # 绘制独立图表
    data = plot_loss_and_validation(log_text, save_path='training_curves.png')

    # 打印训练摘要
    print_training_summary(data)