import torch


def check_pytorch_gpu():
    if torch.cuda.is_available():
        return f"PyTorch can access the GPU: {torch.cuda.get_device_name(0)}"
    else:
        return "PyTorch cannot access the GPU"


# Run the function and print the result
print(check_pytorch_gpu())
