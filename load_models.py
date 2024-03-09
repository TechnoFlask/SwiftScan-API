from keras.models import Model, load_model
from pathlib import Path
from json import loads

json_path = Path("./models/json_metadata")
model_path = Path("./models/keras_models")

models = []


def load():
    for model in sorted(model_path.glob("*.h5")):
        models.append({"model": load_model(model)})
    for i, meta in enumerate(sorted(json_path.glob("*.json"))):
        with open(meta) as file:
            class_names = loads(file.read())["class_names"]
        models[i]["classes"] = class_names
    return models
