import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available.")

print("All CUDA devices:")
devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
device_names = [torch.cuda.get_device_name(d) for d in devices]

print(device_names)
