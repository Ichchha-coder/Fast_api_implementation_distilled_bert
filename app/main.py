from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import predict_sentiment

# Initialize FastAPI app
app = FastAPI()

# Define the input data model
class InputData(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict(input_data: InputData):
    sentiment = predict_sentiment(input_data.text)
    return {"sentiment": sentiment}

# Run the API with Uvicorn if this script is run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


