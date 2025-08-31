import os
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import numpy as np
import json

# --- 1. SET UP PATHS ---
# Make sure you have downloaded the 'ecommerce-products-image-dataset' from Kaggle
# and placed the unzipped folders (jeans, sofa, tshirt, tv) in a 'data' directory
DATA_DIR = 'Data/archive/train/train'
EMBEDDINGS_FILE = 'product_embeddings.json'

# --- 2. MODEL SELECTION AND FEATURE EXTRACTION ---
# Use a pre-trained ResNet50 model as a feature extractor.
# We remove the final classification layer to get the embeddings.
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()
print("Model loaded successfully.")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 3. PROCESS IMAGES AND GENERATE EMBEDDINGS ---
product_embeddings = []
product_metadata = []

for category in os.listdir(DATA_DIR):
    category_path = os.path.join(DATA_DIR, category)
    if not os.path.isdir(category_path):
        continue

    print(f"Processing category: {category}")
    for image_name in os.listdir(category_path):
        if not image_name.endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(category_path, image_name)
        try:
            image = Image.open(image_path).convert('RGB')
            tensor_image = transform(image)
            tensor_image = tensor_image.unsqueeze(0)

            # Generate embedding
            with torch.no_grad():
                embedding = model(tensor_image).squeeze().numpy().tolist()

            # Store metadata and embedding
            product_embeddings.append(embedding)
            relative_image_path = os.path.relpath(image_path, 'Data/archive')
            product_metadata.append({
                'id': os.path.splitext(image_name)[0],
                'name': image_name,
                'category': category,
                'image_path': relative_image_path.replace('\\\\', '/').replace('\\\\', '/')
            })
        except Exception as e:
            print(f"Could not process image {image_path}: {e}")

# --- 4. CREATE A SEARCH INDEX AND SAVE IT ---
# Use NearestNeighbors for efficient search on high-dimensional data.
neighbors = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(product_embeddings)
print("NearestNeighbors index created.")

# Save embeddings and metadata
with open(EMBEDDINGS_FILE, 'w') as f:
    json.dump({
        'metadata': product_metadata,
        'embeddings': product_embeddings
    }, f)
print(f"Embeddings and metadata saved to {EMBEDDINGS_FILE}.")
