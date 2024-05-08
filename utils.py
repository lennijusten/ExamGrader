import json
from pathlib import Path
import os

def load_system_prompt(filepath: Path) -> str:
    with open(filepath, 'r') as file:
        data = json.load(file)
        return data.get("task_description", "")

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def save_dict_to_jsonl(data_dict, jsonl_file_path):
    # Open the file in write mode
    with open(jsonl_file_path, 'w') as file:
        # If 'data_dict' is actually a list of dictionaries:
        for item in data_dict:
            json.dump(item, file)
            file.write('\n')
        # If 'data_dict' is a single dictionary and you want it as a single line:
        # json.dump(data_dict, file)
        # file.write('\n')

def model_factory(model_config, model_library):
    for provider, provider_menu in model_library.items():
        if model_config['model_name'] in provider_menu["models"]:
            api_key_env_var = provider_menu["api_key_env_var"]
            api_key = os.getenv(api_key_env_var)
            if not api_key:
                raise ValueError(
                    f"No API key found for {provider}. Please set the {api_key_env_var} environment variable.")

            # Check if the model has vision capabilities
            vision = model_config['model_name'] in provider_menu["models_with_vision"]

            # Use the full module path from the configuration
            full_module_path = provider_menu["model_module"]
            model_class_name = provider_menu["model_class"]

            # Dynamically import the module and class
            module = __import__(full_module_path, fromlist=[model_class_name])
            model_class = getattr(module, model_class_name)

            # Initialize and return the model instance
            return model_class(api_key=api_key, model_name=model_config["model_name"], vision=vision, system_prompt=model_config["system_prompt"], model_params=model_config["model_params"])

    raise ValueError(f"Model {model_config['model_name']} is not supported.")
