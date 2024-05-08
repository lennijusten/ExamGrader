import argparse
from datetime import datetime
from models.base_model import BaseModel
from tqdm import tqdm
import pandas as pd
from utils import *
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("exam_path", type=Path, help="Filepath to the input exam (JSONL).")
    parser.add_argument("student_config", type=lambda s: [Path(item) for item in s.split(',')],
                        help="Filepath to student config files (JSON), separated by commas (no space).")
    parser.add_argument("--output_path", type=Path,
                        help="Directory to save the processed exams. If not provided, defaults to the exam name with a timestamp.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output.", default=False)
    parser.add_argument("--log_config", action='store_true',
                        help="Log model configurations to output .csv and .jsonl files", default=False)

    args = parser.parse_args()

    # Set default output_path after parsing if it wasn't explicitly provided
    if args.output_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Construct the default output_path based on exam_path and timestamp
        default_output_name = args.exam_path.stem + f"_output_{timestamp}"
        args.output_path = Path('responses') / default_output_name

    return args

def student_take_exam(exam_df: pd.DataFrame, student_model: BaseModel, verbose=False):
    print("Student taking exam...")
    student_history = []

    for _, question in tqdm(exam_df.iterrows()):

            student_payload = student_model.prepare_student_input(question, student_history)
            student_response = student_model.generate_response(student_payload)

            # Add model response and metadata to the question object
            question["student_response"] = student_response["response_text"]
            question["student_response_time"] = datetime.now().isoformat()
            question["student_model_specified"] = student_model.model_name
            question["student_model_used"] = student_response["model"]
            question["student_input_tokens"] = student_response["input_tokens"]
            question["student_output_tokens"] = student_response["output_tokens"]
            question["student_stop_reason"] = student_response["stop_reason"]
            question["student_model_params"] = student_response["model_params"]
            question["student_system_prompt"] = student_response["system_prompt"]

            if verbose:
                tqdm.write(f"\nQuestion {question['index']}:\n{question['question']}\n")
                tqdm.write(f"Student response:\n{question['student_response']}\n")
                tqdm.write("----------------------------------------------------------------------------------\n")

            student_history.append(question)

    return pd.DataFrame(student_history)


if __name__ == "__main__":
    args = parse_arguments()

    # Make output directory; save student and grader config
    args.output_path.mkdir(parents=True)

    # Load exam
    df_exam = pd.read_json(args.exam_path, lines=True)

    # Load library of all models implemented
    model_library_path = 'models/model_library.json'
    MODEL_LIBRARY = load_config(model_library_path)

    # Handle each student config
    for student_config_path in args.student_config:
        STUDENT_CONFIG = load_config(student_config_path)
        student_model = model_factory(STUDENT_CONFIG, MODEL_LIBRARY)
        shutil.copy(student_config_path, args.output_path / student_config_path.name)

        df_exam_responses = student_take_exam(df_exam, student_model, args.verbose)
        df_exam_responses_output_path = args.output_path / f'exam_responses_{student_config_path.stem}.csv'
        df_exam_responses.to_csv(df_exam_responses_output_path, index=False)