import gradio as gr
import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image
from pathlib import Path

# Load checkpoint at startup (change if needed)
CKPT_PATH = Path("outputs/best.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CKPT_PATH, map_location=device)

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

model = build_model(ckpt["model_name"]).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
img_size = ckpt.get("img_size", 224)
class_to_idx = ckpt.get("class_to_idx", {"no_rust": 0, "rust": 1})
idx_to_class = {v:k for k,v in class_to_idx.items()}

tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

def predict(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    x = tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).squeeze(1)
        prob_rust = torch.sigmoid(logits).item()
    pred = 1 if prob_rust >= 0.5 else 0
    label = idx_to_class[pred]
    confidences = {"rust": prob_rust, "no_rust": 1.0 - prob_rust}
    return label, confidences

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Label(label="Confidence")
    ],
    title="Rust Classifier",
    description="Upload an image. The model predicts whether it shows rust."
)

if __name__ == "__main__":
    demo.launch()
