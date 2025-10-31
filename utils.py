import torch
def save_checkpoint(model, optimizer, epoch, path="models/checkpoint.pth"):
    torch.save({"epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()}, path)
    print(f"✅ Checkpoint saved → {path}")

