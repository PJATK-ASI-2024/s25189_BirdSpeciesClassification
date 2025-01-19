import io
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define model class
class BirdResNetModel(nn.Module):
    def __init__(self, num_classes=200):
        super(BirdResNetModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    

# Define FastAPI app
main = FastAPI()

model_path = "../app/model/best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BirdResNetModel(num_classes=200)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


with open("../app/model/class_mappings/updated_class_mapping.json", "r") as f:
    class_mapping = json.load(f)

@main.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_id = torch.argmax(probabilities, dim=1).item()

        # Get class name
        predicted_class_name = class_mapping.get(str(predicted_class_id + 1), "Unknown")

        return JSONResponse({
            "class_id": predicted_class_id + 1,
            "class_name": predicted_class_name
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


