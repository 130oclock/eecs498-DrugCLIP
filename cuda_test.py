import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(device)}")
else:
    print("CUDA is not available.")

print(torch.cuda.current_device())