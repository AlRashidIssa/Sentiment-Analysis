from abc import ABC, abstractmethod
from typing import Optional, Any
from tensorflow.keras.models import load_model  # type: ignore
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import os
import sys


# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.logging_utils import app_logger

# Define tokenizer parameters
MAX_WORDS = 20000  # maximum number of unique words to keep
MAX_SEQUENCE_LENGTH = 200  # maximum length of review sequences

class IPrediction(ABC):
    """
    Interface for making predictions using a pre-trained model and tokenizer.
    """

    @abstractmethod
    def predict(self, model: tf.keras.Model, tokenizer: tf.keras.preprocessing.text.Tokenizer, text: str) -> str:
        """
        Abstract method to predict sentiment from a given text.

        Args:
            model (tf.keras.Model): The pre-trained model used for prediction.
            tokenizer (tf.keras.preprocessing.text.Tokenizer): The tokenizer used to preprocess the text.
            text (str): The input text to predict sentiment on.

        Returns:
            str: The predicted sentiment with confidence score.
        """
        pass

class Prediction(IPrediction):
    """
    Implementation of IPrediction for making sentiment predictions with a Keras model and tokenizer.
    """

    def predict(self, model: tf.keras.Model, tokenizer: tf.keras.preprocessing.text.Tokenizer, text: str) -> str:
        """
        Predicts sentiment from the input text using a pre-trained model and tokenizer.

        Args:
            model (tf.keras.Model): The pre-trained model used for prediction.
            tokenizer (tf.keras.preprocessing.text.Tokenizer): The tokenizer used to preprocess the text.
            text (str): The cleaned input text to predict sentiment on.

        Returns:
            str: The predicted sentiment and confidence score.

        Raises:
            TypeError: If inputs are not of the expected types.
            Exception: For any other errors encountered during prediction.
        """
        try:
            # Check model
            if not isinstance(model, tf.keras.Model):
                raise TypeError("Provided model is not a valid Keras model.")

            # Check tokenizer
            if not isinstance(tokenizer, tf.keras.preprocessing.text.Tokenizer):
                raise TypeError("Provided tokenizer is not a valid Keras tokenizer.")

            # Check text input
            if not isinstance(text, str):
                raise TypeError("Provided text input is not a string.")

            # Convert text to sequence
            sequence = tokenizer.texts_to_sequences([text])
            # Pad the sequence
            padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

            # Predict sentiment
            prediction = model.predict(padded_sequence)

            # Map prediction to sentiment label
            sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
            confidence = prediction[0][0]

            app_logger.info(f"Prediction made successfully for input text.")

            return f"Sentiment: {sentiment}, Confidence: {confidence:.2f}"

        except TypeError as type_err:
            app_logger.error(f"Type error during prediction: {type_err}")
            return "Error: Invalid input type for model, tokenizer, or text."

        except Exception as error:
            app_logger.error(f"An unexpected error occurred during prediction: {error}")
            return "Error: Prediction failed due to an unexpected issue."

from model.load_pre_trained_model import LoadPreTrainedModel
from utils.load_tokenizer import LoadTokenizer
from processing.clean_text import CleanText

if __name__ == "__main__":
    model = LoadPreTrainedModel().call(model_path="/workspaces/Sentiment-Analysis/models/final_sentiment_model.h5")
    tokenizer = LoadTokenizer().load("/workspaces/Sentiment-Analysis/models/tokenizer.pickle")
    clearn_text = CleanText().clean(text="I Head the model@@ it is the Bad Moves and&& i don't whasch the move again??")
    predict = Prediction().predict(model=model, tokenizer=tokenizer, text=clearn_text)

    print(predict)