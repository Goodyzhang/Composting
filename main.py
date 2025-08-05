import torch
import numpy as np

def check_cuda_and_gpus():
    print("CUDA 可用性:", torch.cuda.is_available())
    gpu_count = torch.cuda.device_count()
    print("可用 GPU 数量:", gpu_count)
    for i in range(gpu_count):
        print(f"\nGPU {i}:")
        print("  名称:", torch.cuda.get_device_name(i))
        print("  总内存: {:.2f} GB".format(torch.cuda.get_device_properties(i).total_memory / 1024 ** 3))
        mem_info = torch.cuda.memory_stats(i)
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
        print(f"  已分配内存: {allocated:.2f} GB")
        print(f"  已保留内存: {reserved:.2f} GB")


def check_npy_file(file_path):
    try:
        data = np.load(file_path, allow_pickle=False)
        print(f"文件: {file_path}")
        print(f"数据类型: {data.dtype}")
        print(f"形状: {data.shape}")
        print(f"前5项:\n{data[:5]}\n")
    except FileNotFoundError:
        print(f"未找到文件 {file_path}，请检查目录或尝试运行LSTM-DataWash.py")


if __name__ == "__main__":
    check_cuda_and_gpus()
    check_npy_file('DATA/LSTM-GHG/sequences/X_train.npy')
    check_npy_file('DATA/LSTM-GHG/sequences/Y_train.npy')
    check_npy_file('DATA/LSTM-GHG/sequences/X_test.npy')
    check_npy_file('DATA/LSTM-GHG/sequences/Y_test.npy')
