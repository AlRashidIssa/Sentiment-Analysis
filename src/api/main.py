import os
import sys
from abc import ABC, abstractmethod
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

# Custom functions and methods
from utils.logging_utils import app_logger
from model.load_pre_trained_model import LoadPreTrainedModel
from utils.load_tokenizer import LoadTokenizer
from processing.clean_text import CleanText
from prediction.prediction import Prediction

# Load pre-trained model and initialize FastAPI and templates
MODEL = LoadPreTrainedModel().call(model_path="/workspaces/Sentiment-Analysis/models/final_sentiment_model.h5")
APP = FastAPI()
TEMPLATES = Jinja2Templates(directory="/workspaces/Sentiment-Analysis/src/api/template")

@APP.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    """
    GET method to render the HTML form.
    """
    return TEMPLATES.TemplateResponse("index.html", {"request": request})

class IPipelinePredictionAPI(ABC):
    """
    Interface for defining the pipeline for sentiment prediction.
    """

    @abstractmethod
    def pipeline(self, text: str) -> str:
        """
        Abstract method to define the sentiment prediction pipeline.

        Args:
            text (str): The input text for sentiment analysis.

        Returns:
            str: The prediction result.
        """
        pass

class PipelinePredictionAPI(IPipelinePredictionAPI):
    """
    Implementation of the sentiment prediction pipeline.
    """

    def __init__(self):
        self.model = MODEL
        self.tokenizer = LoadTokenizer().load("/workspaces/Sentiment-Analysis/models/tokenizer.pickle")

    def pipeline(self, text: str) -> str:
        """
        Predict sentiment for the provided text.

        Args:
            text (str): The input text to analyze.

        Returns:
            str: The prediction result or an error message.
        """
        try:
            # Ensure the text input is valid
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Invalid input: text must be a non-empty string.")

            # Clean text
            cleaned_text = CleanText().clean(text=text)
            app_logger.info(f"Cleaned text: {cleaned_text}")

            # Make prediction
            prediction_result = Prediction().predict(model=self.model, tokenizer=self.tokenizer, text=cleaned_text)
            app_logger.info(f"Prediction result: {prediction_result}")
            return prediction_result

        except Exception as e:
            app_logger.error(f"An error occurred in the prediction pipeline: {e}")
            return "An error occurred while processing your request. Please try again."

# Instantiate the PipelinePredictionAPI
pipeline_api = PipelinePredictionAPI()

@APP.post("/predict", response_class=HTMLResponse)
async def predict_sentiment(request: Request, text: str = Form(...)):
    """
    POST method to handle sentiment prediction.

    Args:
        request (Request): The FastAPI request object.
        text (str): The input text to analyze.

    Returns:
        HTMLResponse: The rendered HTML response with prediction results or error messages.
    """
    # Run the prediction pipeline
    prediction_result = pipeline_api.pipeline(text)
    return TEMPLATES.TemplateResponse("index.html", {"request": request, "result": prediction_result, "input_text": text})
