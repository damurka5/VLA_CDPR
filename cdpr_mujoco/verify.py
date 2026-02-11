import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Supported Arch: {torch.cuda.get_arch_list()}")