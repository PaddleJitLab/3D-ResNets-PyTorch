import torch
from models.densenet import generate_model

if __name__ == "__main__":
    x = torch.randn(1, 1, 3, 224, 224)
    model = generate_model(169)
    try:
        torch.export.export(model, (x,))
        print("[JIT] torch.export successed.")
        exit(0)
    except Exception as e:
        print("[JIT] torch.export failed.")
        raise e
