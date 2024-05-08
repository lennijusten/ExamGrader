import anthropic
from models.base_model import BaseModel
from typing import Dict, List, Optional, Union
import warnings
from pathlib import Path

class AnthropicModel(BaseModel):

    def __init__(self, api_key: str, model_name: str, vision: bool = False, system_prompt: Optional[str] = None, model_params: Optional[Dict] = BaseModel):
        super().__init__(api_key, model_name, vision, system_prompt, model_params)


    def truncate_conversation(self) -> None:
        """
        Adjusts the conversation history to fit within the model's token limits.
        """
        pass  # Placeholder for actual truncation logic

    def prepare_student_input(self, question, conversation_history):
        self.truncate_conversation()

        messages = []

        # Add conversation history to messages
        for entry in conversation_history:
            messages.append({"role": "user", "content": entry["question"]})  # Append user prompt to messages
            messages.append({"role": "assistant", "content": entry["student_response"]})  # Append model response to messages

        # Add the current prompt as the latest message from the user
        images = question.get("image", None)
        if not self.vision:
            if images:
                print(f"\nWarning: Question {question['index']} has images but the selected model does not "
                      f"have vision capabilities. Ignoring images and processing text only.")

            messages.append({"role": "user", "content": question['question']})
        else:
            if images:
                content = []
                for image_path in images:
                    encoded_image = self.encode_image(image_path)
                    media_type = f"image/{Path(image_path).suffix[1:]}"  # Determine media type based on file extension
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded_image,
                        }
                    })
                content.append({"type": "text", "text": question['question']})
                # Always append the question text last
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": question['question']})

        return messages

    def prepare_grader_input(self, question, conversation_history):
        self.truncate_conversation()

        def prepare_grader_prompt(question, student_response, answer, points):
            prompt = (f"Question: {question}\n"
                      f"Student response: {student_response}\n"
                      f"Answer key: {answer}\n"
                      f"Total points available: {points}\n")
            return prompt

        messages = []

        # Add conversation history to messages
        for entry in conversation_history:
            prompt = prepare_grader_prompt(entry['question'], entry['student_response'], entry['answer'], entry['points'])
            messages.append({"role": "user", "content": prompt})  # Append user prompt to messages
            messages.append({"role": "assistant", "content": entry["grader_response"]})  # Append model response to messages

        # Add the current prompt as the latest message
        current_prompt = prepare_grader_prompt(question['question'], question['student_response'], question['answer'], question['points'])

        images = question.get("image", None)
        if not self.vision:
            if images:
                print(f"\nWarning: Question {question['index']} has images but the selected model does not "
                      f"have vision capabilities. Ignoring images and processing text only.")

            messages.append({"role": "user", "content": current_prompt})
        else:
            if images:
                content = []
                for image_path in images:
                    encoded_image = self.encode_image(image_path)
                    media_type = f"image/{Path(image_path).suffix[1:]}"  # Determine media type based on file extension
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded_image,
                        }
                    })
                content.append({"type": "text", "text": current_prompt})
                # Always append the question text last
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": current_prompt})

        return messages

    def generate_response(self, messages, verbose=True):
        client = anthropic.Anthropic(api_key=self.api_key)

        params = {
            'model': self.model_name,
            'messages': messages,
            'max_tokens': 3000,  # default max_tokens (required param)
            'temperature': 1.0,  # default
        }

        # Update params with any model-specific configurations
        params.update(self.model_params)

        # Add system prompt if defined
        if self.system_prompt:
            params['system'] = self.system_prompt

        response = client.messages.create(**params)

        if verbose and response.stop_reason != 'end_turn':
            warnings.warn(f"WARNING: Stop reason is {response.stop_reason}")

        response_dict = {}
        response_dict["response_text"] = response.content[0].text
        response_dict["input_tokens"] = response.usage.input_tokens
        response_dict["output_tokens"] = response.usage.output_tokens
        response_dict["stop_reason"] = response.stop_reason
        response_dict["model"] = response.model
        response_dict["model_params"] = self.model_params
        response_dict["system_prompt"] = self.system_prompt

        return response_dict
