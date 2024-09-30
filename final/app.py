import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pickled model and vectorizer
with open('model.pk', 'rb') as model_file:
    X_train, y_train = pickle.load(model_file)

with open('vectorizer.pk', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Create a FastAPI instance
app = FastAPI()

# Define a Pydantic model for the request body
class JobDescription(BaseModel):
    description: str

# Preprocess the job description
def preprocess_description(description):
    # Clean the job description
    description = description.lower().replace(r'[^a-z\s]', '')  # Clean text
    # Vectorize the description
    vectorized_description = vectorizer.transform([description]).toarray()
    return vectorized_description

# Calculate the Euclidean distance
def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))

# Get neighbors based on Euclidean distance
def get_neighbors(X_train, y_train, test_row, num_neighbors):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(test_row, X_train[i])
        distances.append((y_train.iloc[i], dist))
    distances.sort(key=lambda x: x[1])
    neighbors = [distances[i][0] for i in range(num_neighbors)]
    return neighbors

# Make predictions
def predict(X_train, y_train, test_row, num_neighbors):
    neighbors = get_neighbors(X_train, y_train, test_row, num_neighbors)
    prediction = max(set(neighbors), key=neighbors.count)
    return prediction

# Define the prediction endpoint
@app.post("/predict/")
async def predict_job_title(job_desc: JobDescription):
    vectorized_description = preprocess_description(job_desc.description)
    predicted_label = predict(X_train, y_train, vectorized_description[0], num_neighbors=5)
    return {"predicted_work_type": predicted_label}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
