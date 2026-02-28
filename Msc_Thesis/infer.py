import argparse
from pathlib import Path
import torch
from torchvision import transforms, models
from PIL import Image
from torch import nn

def build_model(model_name: str, num_classes: int = 1):
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)
        in_feats = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Identity()
        head = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_feats, num_classes))
        return nn.Sequential(backbone, head)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        return model
    else:
        raise ValueError("Unsupported model")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to best.pt")
    parser.add_argument("--image", type=Path, required=True, help="Path to image to classify")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)

    model = build_model(ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img_size = ckpt.get("img_size", 224)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x).squeeze(1)
        prob_rust = torch.sigmoid(logits).item()

    # Determine label names (fallback if missing)
    class_to_idx = ckpt.get("class_to_idx", {"no_rust": 0, "rust": 1})
    idx_to_class = {v:k for k,v in class_to_idx.items()}

    pred = 1 if prob_rust >= 0.5 else 0
    print(f"Prediction: {idx_to_class[pred]} | P(rust)={prob_rust:.3f}")

if __name__ == "__main__":
    main()
