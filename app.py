<<<<<<< HEAD
import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
from flask import Flask, request, jsonify, send_from_directory
from sklearn.neighbors import NearestNeighbors
import requests
from io import BytesIO

# --- 1. LOAD PRE-COMPUTED DATA ---
# This data was generated in Step 1.
EMBEDDINGS_FILE = 'product_embeddings.json'
DATA_DIR = 'Data/archive'
with open(EMBEDDINGS_FILE, 'r') as f:
    data = json.load(f)
    product_metadata = data['metadata']
    product_embeddings = np.array(data['embeddings'])

# --- 2. SET UP THE FLASK APP ---
app = Flask(__name__, static_folder='static', static_url_path='')

# --- 3. MODEL SELECTION AND FEATURE EXTRACTION ---
# We need to re-load the same model used in the preprocessing step.[1, 2]
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. CREATE A SEARCH INDEX ---
# We fit the NearestNeighbors model again using the loaded embeddings.
neighbors = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(product_embeddings)

# --- 5. IMAGE PROCESSING FUNCTION ---
def get_image_embedding(image):
    """Generates an embedding for a new image."""
    image = image.convert('RGB')
    tensor_image = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor_image).squeeze().numpy()
    return embedding

# --- 6. API ENDPOINT FOR VISUAL SEARCH ---
@app.route('/search', methods=['POST'])
def search():
    try:
        if 'file' in request.files and request.files['file'].filename!= '':
            # Handle file upload.[3]
            image_file = request.files['file']
            image = Image.open(image_file.stream)
        elif 'url' in request.form and request.form['url']!= '':
            # Handle URL input.[4]
            image_url = request.form['url']
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
        else:
            return jsonify({'error': 'No image file or URL provided'}), 400

        # Generate embedding for the new image
        query_embedding = get_image_embedding(image).reshape(1, -1)

        # Find the nearest neighbors
        distances, indices = neighbors.kneighbors(query_embedding)

        results = []
        for i, dist in zip(indices[0], distances[0]):
            metadata = product_metadata[i]
            # Calculate a simple similarity score based on distance
            max_dist = np.max(distances)
            similarity_score = 100 - (dist * 100) / max_dist if max_dist > 0 else 100
            filename = os.path.basename(metadata['image_path'])
            display_name = os.path.splitext(metadata['name'])[0][:8]
            results.append({
                'id': metadata['id'],
                'name': display_name,
                'category': metadata['category'],
                'image_url': f"/images/{filename}",
                'similarity': max(0, similarity_score),
                'description': metadata.get('description', '')
            })

        return jsonify(results)

    except Exception as e:
        import traceback
        print("Error in /search:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# --- 7. SERVE STATIC FILES ---
# Serves the HTML, CSS, and images
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/images/<path:path>')
def send_images(path):
    print(f"Serving image: {path}")
    # This serves the product images from the test/test directory
    base_dir = os.path.join(os.getcwd(), "Data", "archive", "test", "test")
    return send_from_directory(base_dir, path)

if __name__ == '__main__':
=======
import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
from flask import Flask, request, jsonify, send_from_directory
from sklearn.neighbors import NearestNeighbors
import requests
from io import BytesIO

# --- 1. LOAD PRE-COMPUTED DATA ---
# This data was generated in Step 1.
EMBEDDINGS_FILE = 'product_embeddings.json'
DATA_DIR = 'Data/archive'
with open(EMBEDDINGS_FILE, 'r') as f:
    data = json.load(f)
    product_metadata = data['metadata']
    product_embeddings = np.array(data['embeddings'])

# --- 2. SET UP THE FLASK APP ---
app = Flask(__name__, static_folder='static', static_url_path='')

# --- 3. MODEL SELECTION AND FEATURE EXTRACTION ---
# We need to re-load the same model used in the preprocessing step.[1, 2]
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. CREATE A SEARCH INDEX ---
# We fit the NearestNeighbors model again using the loaded embeddings.
neighbors = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(product_embeddings)

# --- 5. IMAGE PROCESSING FUNCTION ---
def get_image_embedding(image):
    """Generates an embedding for a new image."""
    image = image.convert('RGB')
    tensor_image = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor_image).squeeze().numpy()
    return embedding

# --- 6. API ENDPOINT FOR VISUAL SEARCH ---
@app.route('/search', methods=['POST'])
def search():
    try:
        if 'file' in request.files and request.files['file'].filename!= '':
            # Handle file upload.[3]
            image_file = request.files['file']
            image = Image.open(image_file.stream)
        elif 'url' in request.form and request.form['url']!= '':
            # Handle URL input.[4]
            image_url = request.form['url']
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
        else:
            return jsonify({'error': 'No image file or URL provided'}), 400

        # Generate embedding for the new image
        query_embedding = get_image_embedding(image).reshape(1, -1)

        # Find the nearest neighbors
        distances, indices = neighbors.kneighbors(query_embedding)

        results = []
        for i, dist in zip(indices[0], distances[0]):
            metadata = product_metadata[i]
            # Calculate a simple similarity score based on distance
            max_dist = np.max(distances)
            similarity_score = 100 - (dist * 100) / max_dist if max_dist > 0 else 100
            filename = os.path.basename(metadata['image_path'])
            display_name = os.path.splitext(metadata['name'])[0][:8]
            results.append({
                'id': metadata['id'],
                'name': display_name,
                'category': metadata['category'],
                'image_url': f"/images/{filename}",
                'similarity': max(0, similarity_score),
                'description': metadata.get('description', '')
            })

        return jsonify(results)

    except Exception as e:
        import traceback
        print("Error in /search:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# --- 7. SERVE STATIC FILES ---
# Serves the HTML, CSS, and images
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/images/<path:path>')
def send_images(path):
    print(f"Serving image: {path}")
    # This serves the product images from the test/test directory
    base_dir = os.path.join(os.getcwd(), "Data", "archive", "test", "test")
    return send_from_directory(base_dir, path)

if __name__ == '__main__':
>>>>>>> blackboxai/upload-project
    app.run(debug=True)