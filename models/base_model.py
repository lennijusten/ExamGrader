from typing import List, Optional, Union, Dict
import base64

class BaseModel:
    def __init__(self, api_key: str, model_name: str, vision: bool = False, system_prompt: Optional[str] = None, model_params: Optional[Dict] = None):
        self.api_key = api_key
        self.model_name = model_name
        self.vision = vision
        self.system_prompt = system_prompt
        self.model_params = model_params

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def truncate_conversation(self, new_prompt: str, token_limit: Optional[int] = None) -> None:
        """
        Truncate the conversation history to fit within a specified token limit.
        Placeholder method; subclasses should override with model-specific logic.
        """
        raise NotImplementedError

    def prepare_student_input(self):
        """
        Base method for assembling student prompt from history; override in subclasses if needed. Concatenates chat
        history with labels indicating user prompts, models responses, and system messages.
        """
        return NotImplementedError


    def prepare_grader_input(self):
        """
        Base method for assembling grader prompt from history; override in subclasses if needed. Concatenates chat
        history with labels indicating user prompts, models responses, and system messages.
        """
        raise NotImplementedError


    def generate_response(self):
        """
        Generic method for sending payload to model API
        Placeholder method; subclasses should override with model-specific logic.
        """
        raise NotImplementedError
