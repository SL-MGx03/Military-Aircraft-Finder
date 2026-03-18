from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# 1. Define the transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Point to the ROOT folder (the one containing the 6 sub-folders)
train_dataset = datasets.ImageFolder(root='military_aircraft_data_set/Train', 
                                    transform=data_transforms)

test_dataset = datasets.ImageFolder(root='military_aircraft_data_set/Test',
                                    transform=data_transforms)

total = len(train_dataset) + len(test_dataset)

# 4. The Loaders (This is your X and y delivery system)
batch_size=32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_class_names = train_dataset.classes
test_class_names = test_dataset.classes
train_class_to_idx = train_dataset.class_to_idx
test_class_to_idx = test_dataset.class_to_idx

transform_dict = {
    "train_class_names" :train_class_names,
    "train_class_to_idx" :train_class_to_idx,
    "test_class_names" :test_class_names,
    "test_class_to_idx" :test_class_to_idx,
    "DataLoader" :DataLoader,
    "train_loader":train_loader,
    "test_loader" :test_loader,
    "batch_size" :batch_size
}


def check_train_test():
    print(f"Train images found {len(train_dataset)} | Test Images Found {len(test_dataset)} |Total images found: {total}")
    print(f"Classes detected: {train_class_names}")

    if train_class_names == test_class_names:
        print("train and test class names are equal")

    else:
        print("train and test class names are not equal")

