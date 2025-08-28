import torch

print("Torch version:", getattr(torch, '__version__', None))
# torch.version.cuda may not exist in some builds; use getattr to avoid static analysis warnings
print("Torch cuda build:", getattr(getattr(torch, 'version', None), 'cuda', None))
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        try:
            caps = torch.cuda.get_device_capability(i)
        except Exception:
            caps = None
        try:
            name = torch.cuda.get_device_name(i)
        except Exception:
            name = '<unknown>'
        print(f"  {i}: {name} (capability {caps})")
    try:
        x = torch.rand(5000, 5000, device="cuda")
        y = torch.mm(x, x)
        print("GPU test successful, result shape:", y.shape)
    except RuntimeError as e:
        print("GPU runtime error:", type(e).__name__, e)
        print("Possible cause: your GPU requires a PyTorch build that supports its CUDA compute capability.")
        print("If you have an NVIDIA RTX 5080 (compute capability 12.0+), install a nightly cu128 wheel. Example (PowerShell):")
        print()
        print("```powershell")
        print("pip uninstall -y torch torchvision torchaudio")
        print("python -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio")
        print("```")
        print()
        print("After installation, reactivate your venv and re-run this script.")
else:
    print("No CUDA device detected")
