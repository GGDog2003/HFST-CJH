def calculate_average_metrics_from_file(file_path):
    """
    从日志文件中读取内容，自动提取所有PSNR和SSIM，计算平均值
    """
    psnr_list = []
    ssim_list = []

    try:
        # 打开文件并逐行读取
        with open(file_path, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()

        # 逐行解析
        for line in log_lines:
            line = line.strip()
            # 匹配日志格式：| 31.24dB 0.9628
            if "dB" in line and "|" in line:
                try:
                    parts = line.split("|")[-1].strip()
                    psnr_str = parts.split("dB")[0].strip()
                    ssim_str = parts.split()[-1].strip()

                    psnr = float(psnr_str)
                    ssim = float(ssim_str)

                    psnr_list.append(psnr)
                    ssim_list.append(ssim)
                except:
                    continue

        # 计算结果
        if len(psnr_list) == 0:
            print("\n未检测到有效数据！")
            return

        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)

        # 输出
        print("\n" + "=" * 60)
        print(f"✅ 共检测到 {len(psnr_list)} 个测试样本")
        print(f"📊 平均 PSNR: {avg_psnr:.4f} dB")
        print(f"📊 平均 SSIM: {avg_ssim:.4f}")
        print("=" * 60)

    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
    except Exception as e:
        print(f"读取文件出错：{e}")


if __name__ == "__main__":
    # ========== 在这里修改你的日志文件路径 ==========
    log_file = "tmp_test_log.txt"
    calculate_average_metrics_from_file(log_file)