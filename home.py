from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import pickle
from obj_detection import ObjDetection
from src.utilities import ExactIndex, extract_img, similar_img_search
from flask_cors import CORS
from torchvision import transforms
import os

app = Flask(__name__)
CORS(app)

yolo = ObjDetection(onnx_model='./models/best.onnx', data_yaml='./models/data.yaml')

with open("img_paths.pkl", "rb") as im_file:
    image_paths = pickle.load(im_file)

with open("embeddings.pkl", "rb") as file:
    embeddings = pickle.load(file)

loaded_idx = ExactIndex.load(embeddings, image_paths, "flatIndex.index")

transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image_obj = Image.open(image_file)
    image_array = np.array(image_obj)
    cropped_objs = yolo.crop_objects(image_array)

    recommendations = []
    if cropped_objs is not None:
        for obj in cropped_objs:
            embedding = extract_img(obj, transformations)
            selected_neighbor_paths = similar_img_search(embedding, loaded_idx)
            recommendations.extend(selected_neighbor_paths)

    # Limitar as recomendações com base na entrada do usuário
    num_recommendations = int(request.form.get('numRecommendations', 1))  # padrão é 1
    recommendations = recommendations[:num_recommendations]  # limitar as recomendações

    # Construir URLs
    recommendations_urls = [f"http://localhost:5000/index_images/{os.path.basename(path)}" for path in recommendations]
    return jsonify({"recommendations": recommendations_urls})

@app.route('/index_images/<path:filename>', methods=['GET'])
def serve_image(filename):
    return send_from_directory('index_images', filename)

if __name__ == '__main__':
    app.run(debug=True)
