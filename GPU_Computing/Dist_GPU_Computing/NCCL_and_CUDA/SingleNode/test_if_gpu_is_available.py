import torch

if torch.cuda.is_available():
    print("CUDA is available!")
    # Optionally, you can also print the number of GPUs and their names
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. PyTorch will use the CPU.")


