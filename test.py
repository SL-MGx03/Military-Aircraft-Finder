import torch
from PIL import Image
import torchvision.models as models
from torch import nn
from data_transforms import data_transforms, transform_dict

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Re-create the model structure
model_path = "military_aircraft_model.pth"
model = models.mobilenet_v3_small(weights=None)

num_features = model.classifier[3].in_features
class_names = transform_dict["train_class_names"]
model.classifier[3] = nn.Linear(num_features, len(class_names))

# 3. Load the saved weights into the model
model_path = "military_aircraft_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def predict_on_image(image_path, model, transform, class_names, device):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(dim=0).to(device)
    
    with torch.inference_mode():
        logits = model(img_tensor)
        
    probs = torch.softmax(logits, dim=1)
    pred_label = torch.argmax(probs, dim=1).item()
    conf_score = torch.max(probs).item()
    
    return class_names[pred_label], conf_score

# 4. Run the prediction
custom_image_path = r" "# Enter image path

try:
    label, confidence = predict_on_image(custom_image_path, model, data_transforms, class_names, device)
    print(f"\n--- Prediction Result ---")
    print(f"Military Aircraft Type: {label}")
    print(f"Confidence: {confidence*100:.2f}%")
except FileNotFoundError:
    print(f"Error: Could not find the image at {custom_image_path}")