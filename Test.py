import argparse, torch
from Model import VGG6
from DataModel import get_dataloaders

def accuracy_top1(logits, targets):
    with torch.no_grad():
        return (logits.argmax(dim=1)==targets).float().mean().item()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); tot=n=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        tot += accuracy_top1(model(x), y) * y.size(0); n += y.size(0)
    return tot/n

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)

    blob = torch.load(args.ckpt, map_location=device)
    activation = blob.get("cfg", {}).get("activation", "relu")

    model = VGG6(activation=activation).to(device)
    model.load_state_dict(blob["model_state"], strict=True)

    test_acc = evaluate(model, test_loader, device)
    print(f"[TEST ONLY] Stored-model CIFAR-10 accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
