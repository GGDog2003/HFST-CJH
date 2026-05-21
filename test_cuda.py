import torch
print(torch.cuda.is_available())  # 现在应该输出 True
print(torch.version.cuda)         # 查看CUDA版本
print(torch.__version__)