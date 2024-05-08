import argparse
from datetime import datetime
from utils import *
from tqdm import tqdm
import warnings
from models.base_model import BaseModel
import pandas as pd
import numpy as np
import shutil
import ast

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("exam_path", type=Path, help="Filepath to the input exam (JSONL).")
    parser.add_argument("grader_config", type=Path,
                        help="Filepath to the grader model config (JSON), required if grading is enabled.")
    parser.add_argument("--output_path", type=Path,
                        help="Directory to save the processed exams. If not provided, defaults to the exam name with a timestamp.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output.", default=False)
    parser.add_argument("--log_config", action='store_true', help="Log model configurations to output .csv and .jsonl files", default=False)


    args = parser.parse_args()

    # Set default output_path after parsing if it wasn't explicitly provided
    if args.output_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Construct the default output_path based on exam_path and timestamp
        default_output_name = args.exam_path.stem + f"_output_{timestamp}"
        args.output_path = Path('responses') / default_output_name

    return args


def grader_grade_exam(exam_df: str, grader_model: BaseModel, verbose=False):
    """
    The grader model is expected to output a JSON-formatted string with the keys 'grader_score' and 'grader_justification'.
    If the JSON containing these keys is not included in the grader output, the script will raise a warning and append
    null responses for those keys in the grader_history. It will still record the grader_response.
    These instructions should be included in the system prompt. For OpenAI models, the JSON format can be enforced using
    the response_format argument.
    """

    print("Grader grading exam...")
    grader_history = []

    for _, question in tqdm(exam_df.iterrows()):

            grader_payload = grader_model.prepare_grader_input(question, grader_history)
            grader_response = grader_model.generate_response(grader_payload)

            question["grader_response"] = grader_response["response_text"]
            question["grader_score"] = np.nan
            question["grader_justification"] = ""

            try:
                json_response = json.loads(question["grader_response"])
            except json.JSONDecodeError:
                warnings.warn(f"Failed to decode JSON from grader response text for question [{question['index']}].")
                json_response = None

            if json_response:
                try:
                    question["grader_score"] = json_response['grader_score']
                    question["grader_justification"] = json_response["grader_justification"]
                except KeyError as e:
                    warnings.warn(f"Successfully decoded JSON from grader response text for question [{question['index']}] but missing key: {e}")

            question["grader_response_time"] = datetime.now().isoformat()
            question["grader_model_specified"] = grader_model.model_name
            question["grader_model_used"] = grader_response["model"]
            question["grader_input_tokens"] = grader_response["input_tokens"]
            question["grader_output_tokens"] = grader_response["output_tokens"]
            question["grader_stop_reason"] = grader_response["stop_reason"]
            question["grader_model_params"] = grader_response["model_params"]
            question["grader_system_prompt"] = grader_response["system_prompt"]

            if verbose:
                tqdm.write(f"\nQuestion {question['index']}:\n{question['question']}\n")
                tqdm.write(f"Student answer:\n{question['student_response']}\n")
                tqdm.write(f"Grader score: {question['grader_score']}/{question['points']}\n"
                      f"Grader justification: {question['grader_justification']}\n"
                      f"----------------------------------------------------------------------------------\n")

            grader_history.append(question)

    return pd.DataFrame(grader_history)


if __name__ == "__main__":
    args = parse_arguments()

    # Make output directory; save student and grader config
    args.output_path.mkdir(parents=True)

    # Load exam
    df_exam = pd.read_csv(args.exam_path)
    df_exam['image'] = df_exam['image'].apply(lambda x: ast.literal_eval(x))  # interpret image col as list

    # Load library of all models implemented
    model_library_path = 'models/model_library.json'
    MODEL_LIBRARY = load_config(model_library_path)

    # Load grader model
    GRADER_CONFIG = load_config(args.grader_config)
    grader_model = model_factory(GRADER_CONFIG, MODEL_LIBRARY)

    shutil.copy(args.grader_config, args.output_path / 'grader.json')
    df_graded_exam = grader_grade_exam(df_exam, grader_model, args.verbose)
    df_graded_exam_output_path = args.output_path / 'graded_exam.csv'
    df_graded_exam.to_csv(df_graded_exam_output_path, index=False)