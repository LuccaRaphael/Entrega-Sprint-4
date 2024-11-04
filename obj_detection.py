import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader

class ObjDetection():
    def __init__(self, onnx_model, data_yaml):
        # Carrega o arquivo yaml com dados
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']

        # Carrega o modelo de detecção de objetos
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def crop_objects(self, image):
        row, col, d = image.shape

        # Converte a imagem para uma matriz quadrada
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        # Obtém a previsão do modelo usando a imagem quadrada
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward() # Previsão do modelo

        # NMS
        # Filtra detecções com base nos valores de confiança e probabilidade (0.1)
        # Inicializa variáveis
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        # Define a largura e altura da imagem (input_image)
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4] # Confiança na detecção do objeto
            if confidence > 0.10:
                class_score = row[5:].max() # Maior probabilidade entre os objetos
                class_id = row[5:].argmax() # Índice do objeto com maior probabilidade

                if class_score > 0.10:
                    cx, cy, w, h = row[0:4]
                    
                    # Cria a caixa delimitadora (bbox) a partir dos 4 valores
                    # esquerda, topo, largura e altura
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])

                    # Adiciona os valores nas listas
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # Organiza os resultados
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # Aplica NMS
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.10, 0.10).flatten()

        # Obtém as caixas delimitadoras e recorta os objetos da imagem
        cropped_objects = []
        for ind in index:
            x, y, w, h = boxes_np[ind]
            x1 = int(x)
            y1 = int(y)
            x2 = int((x + w))
            y2 = int((y + h))

            # Recorta o objeto da imagem
            cropped_obj = image[y1:y2, x1:x2].copy()
            cropped_objects.append(cropped_obj)

        return cropped_objects
