import pandas as pd
import os
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .utils import extract_features
from .utils import model, label_encoder
import numpy as np

# API 1: COLLECT COORDINATES
@api_view(['POST'])
def collect_coordinates(request):
    coords = request.data.get("coordinates")
    label = request.data.get("label")

    if not coords or not label:
        return Response({"error": "Invalid data"}, status=400)

    row = coords + [label]
    df = pd.DataFrame([row])

    filename = "double_hand_dataset.csv"
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', index=False, header=False)
    else:
        df.to_csv(filename, index=False)

    return Response({"message": "Coordinates saved"})

# API 2: TRAIN MODEL
@api_view(['POST'])
def train_model_api(request):
    os.system("python train_model_from_csv.py")
    return Response({"message": "Model trained successfully"})
