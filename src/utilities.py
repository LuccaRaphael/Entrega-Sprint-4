import torch
import gc
import faiss
import random

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from featurizer_model import FeaturizerModel

class ExactIndex():
    def __init__(self, vectors, img_paths):
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype('float32')
        self.img_paths = img_paths

    def build(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.vectors)

    def query(self, vectors, k=10):
        distances, indices = self.index.search(vectors, k)
        return [self.img_paths[i] for i in indices[0]]
    
    def save(self, filename):
        faiss.write_index(self.index, filename)
    
    @classmethod
    def load(cls, vectors, img_paths, filename):
        instance = cls(vectors, img_paths)
        instance.index = faiss.read_index(filename)
        return instance

def display_image(file_name):
    try:
        img = Image.open(file_name)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Ocorreu um erro: {e}")


def visualize_nearest_neighbors(selected_img_path, nearest_neighbor_paths):
    # Cria uma figura com duas colunas
    fig, axs = plt.subplots(5, 2, figsize=(10, 8))

    plt.suptitle("Itens Recomendados com base na sua seleção", fontsize=16, y=1.03)

    # Exibe o item selecionado na primeira coluna (coluna 0)
    selected_img = mpimg.imread(selected_img_path)
    axs[0, 0].imshow(selected_img)
    axs[0, 0].set_title("Item selecionado")
    axs[0, 0].axis('off')

    # Limita o número de vizinhos exibidos a um máximo de 10
    num_neighbors = min(len(nearest_neighbor_paths), 10)

    # Percorre os itens recomendados (vizinhos mais próximos) e exibe na segunda coluna (coluna 1)
    for i, ax in enumerate(axs[1:].flatten(), 1):
        if i <= num_neighbors:
            neighbor_path = nearest_neighbor_paths[i - 1]
            img = mpimg.imread(neighbor_path)
            ax.imshow(img)
            ax.set_title(f"Item Recomendado {i}")
            ax.axis('off')

    # Oculta a linha do eixo na segunda coluna da primeira linha
    for i in range(5):
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')

    # Exibe as imagens
    return fig

def extract_img(image, transformation):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FeaturizerModel().to(device)
    model.load_state_dict(torch.load('featurizer-model-1.pt', map_location=device)['model_state_dict'], strict=False)
    model.eval()

    latent_feature = []
    pil_image = Image.fromarray(image)
    tensor = transformation(pil_image.convert("RGB").resize((128, 128))).to(device)
    latent_feature = model.encoder(tensor.unsqueeze(0)).cpu().detach().numpy()
    del tensor
    gc.collect()
    return np.array(latent_feature)

def similar_img_search(query_vector, index):
    query_vector = query_vector.reshape(1, -1)
    nearest_neighbors = index.query(query_vector, k=6)
    selected_neighbors_paths = nearest_neighbors[1:]
    return selected_neighbors_paths

def visualize_outfits(boards):
    # Cria uma figura com duas colunas
    fig, axs = plt.subplots(4, 2, figsize=(10, 8))

    plt.suptitle("Itens recomendados com base nos objetos de moda detectados", fontsize=14, y=1)

    # Limita o número de vizinhos exibidos a um máximo de 6
    num_neighbors = min(len(boards), 6)
    
    # Seleciona aleatoriamente 6 caminhos para exibição
    random_paths = random.sample(boards, num_neighbors)

    # Percorre os itens recomendados e exibe
    for i, ax in enumerate(axs.flatten()):
        if i < len(random_paths):
            neigbor_path = random_paths[i]
            img = mpimg.imread(neigbor_path)
            ax.imshow(img)
            ax.axis('off')

    # Oculta a linha do eixo em todos os eixos
    for i in range(4):
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')

    return fig

def viz_thumbnail(im_path, tn_sz):
    a_img = mpimg.imread(im_path)
    # Obtém as dimensões da imagem original
    img_height, img_width, _ = a_img.shape

    # Calcula o preenchimento necessário para tornar a imagem quadrada
    max_dim = max(img_height, img_width)
    pad_vert = (max_dim - img_height) // 2
    pad_horiz = (max_dim - img_width) // 2

    # Cria uma nova imagem com preenchimento
    padded_img = np.pad(a_img, ((pad_vert, pad_vert), (pad_horiz, pad_horiz), (0, 0)), mode='constant', constant_values=255)

    # Cria a figura e o eixo
    fig, ax = plt.subplots(figsize=tn_sz)

    ax.imshow(padded_img)

    # remove as marcas e rótulos dos eixos para uma aparência mais limpa
    ax.axis('off')

    return fig
