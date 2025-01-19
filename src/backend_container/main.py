import io
import json
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class BirdResNetModel(nn.Module):
    def __init__(self, num_classes=200):
        super(BirdResNetModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

main = FastAPI()

MODE = os.environ.get("RUN_MODE", "local")
if MODE == "container":
    base_dir = "/app"
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, "model/best_model.pth")
class_mapping_path = os.path.join(base_dir, "model/class_mappings/updated_class_mapping.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BirdResNetModel(num_classes=200)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    with open(class_mapping_path, "r") as f:
        class_mapping = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load class mapping: {e}")

@main.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_id = torch.argmax(probabilities, dim=1).item()

        predicted_class_name = class_mapping.get(str(predicted_class_id + 1), "Unknown")

        return JSONResponse({
            "class_id": predicted_class_id + 1,
            "class_name": predicted_class_name
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
