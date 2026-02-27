import torch, torch.nn as nn
from PIL import Image
from torchvision import transforms
import sys

CKPT_PATH = "models/brain_ct_custom_cnn_best.pth"
IMG_SIZE  = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BrainCTCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=2,dilation=2), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,3,padding=2,dilation=2), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(256,128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128,2)
        )
    def forward(self,x):
        x=self.conv1(x); x=self.conv2(x); x=self.conv3(x); x=self.conv4(x)
        x=x.view(x.size(0),-1)
        return self.head(x)

ckpt = torch.load(CKPT_PATH, map_location=device)
classes = ckpt["classes"]; img_size = ckpt.get("img_size", IMG_SIZE)

tfm = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

model = BrainCTCNN(num_classes=len(classes)).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

img_path = sys.argv[1] if len(sys.argv) > 1 else None
assert img_path, "Usage: python brain_ct_infer.py path/to/image.png"

img = Image.open(img_path).convert("L")
x = tfm(img).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
pred_idx = probs.argmax()
print(f"Prediction: {classes[pred_idx]}  |  probs={dict(zip(classes, [float(p) for p in probs]))}")
