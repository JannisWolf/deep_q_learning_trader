'''
Created on 15.11.2017

Utility functions

@author: jtymoszuk
'''
import os
from directories import ROOT_DIR
from keras.models import Sequential
from keras.models import model_from_json
from framework.logger import logger
from directories import ROOT_DIR, DATASETS_DIR


def save_keras_sequential(model: Sequential, relative_path: str, file_name_without_extension: str) -> bool:
    """
    Saves a Keras Sequential in File System
    
    Args:
        model : Sequential to save
        relative_path : relative path in project
        file_name_without_extension : file name without extension, will be used for json with models and h5 with weights.
    Returns:
        True if successful, False otherwise, never None
    """
    if model.model is None:
        logger.error(f"save_keras_sequential: Cannot write an empty model as file")
        return False

    try:
        model_as_json = model.to_json()

        model_filename_with_path = os.path.join(ROOT_DIR, relative_path, file_name_without_extension + '.json')
        weights_filename_with_path = os.path.join(ROOT_DIR, relative_path, file_name_without_extension + '.h5')

        json_file = open(model_filename_with_path, "w")
        json_file.write(model_as_json)
        json_file.close()

        model.save_weights(weights_filename_with_path)
        logger.debug(f"save_keras_sequential: Saved Sequential from {model_filename_with_path} "
                    f"and {weights_filename_with_path}!")
        return True
    except:
        logger.error(f"save_keras_sequential: Writing of Sequential as file failed")
        return False


def load_keras_sequential(relative_path: str, file_name_without_extension: str) -> Sequential:
    """
    Loads a Keras Sequential neural network from file system
    
    Args:
        relative_path : relative path in project
        file_name_without_extension : file name without extension, will be used for json with models and h5 with weights.
    Returns:
        Sequential, or None if nothing found or error
    """

    model_filename_with_path = os.path.join(ROOT_DIR, relative_path, file_name_without_extension + '.json')
    weights_filename_with_path = os.path.join(ROOT_DIR, relative_path, file_name_without_extension + '.h5')

    if os.path.exists(model_filename_with_path) and os.path.exists(weights_filename_with_path):
        try:
            json_file = open(model_filename_with_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(weights_filename_with_path)
            logger.debug(f"load_keras_sequential: Loaded Sequential from {model_filename_with_path} "
                        f"and {weights_filename_with_path}!")
            return model
        except:
            logger.error(f"load_keras_sequential: Loading of Sequential {model_filename_with_path} failed!")
            return None
    else:
        logger.error(f"load_keras_sequential: model File {model_filename_with_path} "
                     f"or weights file {weights_filename_with_path} not found!")
        return None
