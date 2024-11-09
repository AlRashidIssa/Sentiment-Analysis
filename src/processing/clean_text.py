import os
import sys
# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)
import re
import nltk
from abc import ABC, abstractmethod
from typing import Any, Union
from nltk.corpus import stopwords
from utils.logging_utils import app_logger

# Download necessary nltk data
nltk.download("stopwords")
nltk.download("punkt")

# Load stopwords
STOP_WORDS = set(stopwords.words('english'))

class ICleanText(ABC):
    """
    Interface for cleaning text data.
    """

    @abstractmethod
    def clean(self, text: str) -> Union[str, None]:
        """
        Abstract method to clean the input text by removing unwanted characters and stopwords.

        Args:
            text (str): The text to be cleaned.

        Returns:
            Union[str, None]: The cleaned text string or None if cleaning fails.
        """
        pass

class CleanText(ICleanText):
    """
    Concrete implementation of the ICleanText interface for cleaning text data.
    """

    def clean(self, text: str) -> Union[str, None]:
        """
        Cleans the input text by removing special characters, numbers, and stopwords.

        Args:
            text (str): The text to be cleaned.

        Returns:
            Union[str, None]: The cleaned text string if successful, or None if an error occurs.

        Raises:
            ValueError: If the input text is not a string.
        """
        try:
            if not isinstance(text, str):
                raise ValueError("Input text must be a string.")
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-z\s]', '', text)
            
            # Remove stopwords
            words = text.split()
            words = [word for word in words if word not in STOP_WORDS]
            
            # Join the words back into a single string
            cleaned_text = ' '.join(words)
            app_logger.info("Clearing Text!")
            return cleaned_text
        
        except ValueError as ve:
            app_logger.error(f"ValueError: {ve}")
            return None
        except Exception as e:
            app_logger.error(f"An error occurred during text cleaning: {e}")
            return None
        

if __name__ == "__main__":
    iclearn = CleanText()
    clearn_text = iclearn.clean("@@333Hello worlds .///#4 def call inging.. !!! I Love Salah and Naseerasdfghjkl ")
    print("Clearning Text", clearn_text)