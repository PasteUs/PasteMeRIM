from tensorflow import keras
from config import *


def get_model(embedding_layer: keras.layers.Layer) -> keras.models.Model:  # TODO: get embedding layer
    model = keras.models.load_model(MODEL_SAVING_PATH)
    model_with_embedding = keras.Sequential([
        embedding_layer,
        model
    ])

    return model_with_embedding
