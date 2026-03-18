import torch
import torchvision
import torchvision.models as models
from torch import nn
from PIL import Image

from helper_functions import accuracy_fn
from data_transforms import transform_dict, data_transforms

# timer 
from tqdm.auto import tqdm
from timeit import default_timer as timer
train_time_start = timer()

# device diagonistic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# model : mobilenet_v3_small
model_1 = models.mobilenet_v3_small(weights="DEFAULT")

for param in model_1.parameters():
    param.requires_grad = True

# Update the classifier head to output 81 classes
num_features = model_1.classifier[3].in_features
model_1.classifier[3] = nn.Linear(num_features, len(transform_dict["train_class_names"]))

# 3. Move it to the device
model_1 = model_1.to(device)

# loss function
loss_fn = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(params=model_1.parameters(), 
                             lr=0.0001)

model_dict = {
    "Model" : model_1,
    "Loss Function": loss_fn,
    "Optimizer" : optimizer
}

# ----------------------------------------------------------------------------------------

# train function

def train_step(model, transform_dict, loss_fn, optimizer, accuracy_fn, device):
    train_loss , train_acc = 0, 0
    train_loader = transform_dict["train_loader"]
    model.train()

    for batch_idx, (X,y) in enumerate(tqdm(train_loader, desc="  Training")):

        # tensors move to device
        X = X.to(device)
        y = y.to(device)

        train_pred = model(X)
        loss = loss_fn(train_pred,y)
        train_loss += loss.item()
        train_acc += accuracy_fn(y_true=y, y_pred=train_pred.argmax(dim=1)) # train accuracy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    print(f"  Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
    return train_loss, train_acc


# ----------------------------------------------------------------------------------------

# test function

def test_step(model, transform_dict, loss_fn, optimizer, accuracy_fn, device):
    test_loss, test_acc = 0, 0
    test_loader = transform_dict["test_loader"]

    model.eval()
    with torch.inference_mode():
        for (X_test,y_test) in test_loader:

            # tensors to device
            X_test, y_test = X_test.to(device), y_test.to(device)

            test_pred = model(X_test)
            test_loss += loss_fn(test_pred,y_test)
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1)) # testing accuracy

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

    print(f"  Test Loss:  {test_loss:.4f} | Test Accuracy:  {test_acc:.4f}")
    return test_loss, test_acc


# ----------------------------------------------------------------------------------------

# guessing the helicopter type function
 
def predict_on_image(image_path, model, transform, class_names, device):
    # 1. Load image
    img = Image.open(image_path).convert("RGB")
    
    # 2. Transform and add a "Batch" dimension [1, 3, 224, 224]
    img_tensor = transform(img).unsqueeze(dim=0).to(device)
    
    # 3. Predict
    model.eval()
    with torch.inference_mode():
        logits = model(img_tensor)
        
    # 4. Convert raw numbers (logits) to probabilities and find the best one
    probs = torch.softmax(logits, dim=1)
    pred_label = torch.argmax(probs, dim=1).item()
    conf_score = torch.max(probs).item()
    
    return class_names[pred_label], conf_score


# ----------------------------------------------------------------------------------------

epochs =20 # epochs count
best_test_acc = 0

for epoch in range(epochs):

    print(f"Epoch: {epoch+1} \n--------")
    model_dict["Train Loss"], model_dict["Train Accuracy"] = train_step(model_1, transform_dict, loss_fn, optimizer, accuracy_fn, device)
    model_dict["Test Loss"], model_dict["Test Accuracy"] = test_step(model_1, transform_dict, loss_fn, optimizer, accuracy_fn, device)

    current_test_acc = model_dict["Test Accuracy"]
    
    # Save the model weights
    if current_test_acc > best_test_acc:
        best_test_acc = current_test_acc
        torch.save(obj=model_1.state_dict(), f="military_aircraft_model.pth")
        print(f"🌟 New Best Test Accuracy: {best_test_acc:.4f}! Model saved.")

train_time_end = timer()
print(f"Total training time: {train_time_end - train_time_start:.3f} seconds")


# Path to your test image
custom_image_path = r" "# Enter image path

# Use the function
label, confidence = predict_on_image(custom_image_path, 
                                     model_1, 
                                     data_transforms, 
                                     transform_dict["train_class_names"], 
                                     device)

print(f"Prediction: {label} ({confidence*100:.2f}%)")
