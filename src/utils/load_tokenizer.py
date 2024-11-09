import os
import sys
# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)
import pickle
from typing import Any, Union
from abc import ABC, abstractmethod
from utils.logging_utils import app_logger


class ILoadTokenizer(ABC):
    """
    Interface for loading a tokenizer file in pickle format.
    """

    @abstractmethod
    def load(self, path_file: str) -> Union[Any, None]:
        """
        Abstract method to load a tokenizer from a pickle file.

        Args:
            path_file (str): The path to the pickle file containing the tokenizer.

        Returns:
            Union[Any, None]: The loaded tokenizer object, or None if loading fails.
        """
        pass

class LoadTokenizer(ILoadTokenizer):
    """
    Concreate implementation of the ILoadTokenizer iterface for loading a tokenizer from a pickle file.
    """
    
    def load(self, path_file: str) -> Union[Any, None]:
        """
        Loads a tokenizer from a specified pickle file.

        Args:
            path_file (str): The path to the pickle file containing the tokenizer.

        Returns:
            Union[Any, None]: The loaded tokenizer object if successful, or None if an error occurs.

        Raises:
            FileNotFounError: If the specified pickle file does not exist.
            Execption: For any other errors encountered during loadin.
        """
        try:
            with open(path_file, 'rb') as file:
                tokenizer = pickle.load(file)
            app_logger.info(f"Tokenizer loaded successfully from {path_file}")
            return tokenizer
        
        except FileNotFoundError as fnf_error:
            app_logger.error(f"File not found: {path_file} - {fnf_error}")
            return None

        except pickle.UnpicklingError as pickle_error:
            app_logger.error(f"Failed to unpickle file: {path_file} - {pickle_error}")
            return None

        except Exception as error:
            app_logger.error(f"An error occurred while loading the tokenizer from {path_file}: {error}")
            return None

if __name__ == "__main__":
    laod_token = LoadTokenizer()
    tokenizer = laod_token.load("/workspaces/Sentiment-Analysis/models/tokenizer.pickle")
    print("Data Type:", type(tokenizer))
