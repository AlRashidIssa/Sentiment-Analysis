from abc import ABC, abstractmethod
from typing import Optional, Any
from tensorflow.keras.models import load_model #type: ignore
import tensorflow as tf
import os
import sys

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.logging_utils import app_logger

print("TensorFlow Version:", tf.version.VERSION)

class ILoadPreTrainedModel(ABC):
    """
    Interface for loading pre-trained models. Classes implementing this interface
    should define a method for loading a model from a specified path.
    """
    @abstractmethod
    def call(self, model_path: str) -> tf.keras.Model:
        """
        Abstract method to load a mdoel from the given path.

        Parameters:
        - model_path (str): The path to the saved model file.

        Returns:
        - tf.keras.Model: The loaded keras model.
        """
        pass

class LoadPreTrainedModel(ILoadPreTrainedModel):
    """
    Implementation of the ILoadPreTrainedModel interface. Loads a pre-trained 
    TensorFlow keras model from a specified file path with error handling.
    """

    def call(self, model_path: str) -> tf.keras.Model:
        """
        Loads pre-trained model from the specified path, handling errors 
        and logging any issues encountered during the loagging process.

        Parameters:
        - model_path (str): The loaded keras model.

        Raises:
        - FileNotFoundError: If the specified model path does not exist.
        - ValueError: If the loaded model is None.
        - OSError: If there is an issue loading the model file.  
        """
        # Check if the model path exists
        if not os.path.exists(model_path):
            app_logger.error(f"The file does not exist: {model_path}")
            raise FileNotFoundError(f"The specified model path does not exist: {model_path}")
        
        try:
            # Attempt to loead the model
            model = load_model(model_path)

            # Check if the model loaded is None
            if model is None:
                app_logger.error("The Model is None")
                raise ValueError("The loaded model is None. Please check the model file.")
            app_logger.info(f"Model loaded successfully from {model_path}")
            return model
        except (OSError, IOError) as e:
            # Handle file I/O errors
            app_logger.error(f"Error loading model file: {model_path}. Exception: {str(e)}")
            raise OSError(f"Error loading model file: {model_path}") from e
        
        except Exception as e:
            # Catch any other exceptions that may occur
            app_logger.error(f"An unexpected error occurred while loading the model: {str(e)}")
            raise


# Test the Class and main Funcation 
if __name__ == "__main__":
    iload = LoadPreTrainedModel()
    model = iload.call("/workspaces/Sentiment-Analysis/models/final_sentiment_model.h5")