import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load the model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

app = FastAPI()

# Define input data model
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: IrisData):
    input_data = np.array([
        data.sepal_length, data.sepal_width, data.petal_length, data.petal_width
    ]).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    
    species_map = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
    return {"prediction": species_map[prediction]}