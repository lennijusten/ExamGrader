jsonl_file_path = 'exams/7.26-Exam1-2024.jsonl'

total_points = 0

# Open the JSONL file and iterate through each line
with open(jsonl_file_path, 'r') as file:
    for line in file:
        # Convert each line (which is in JSON format) into a Python dictionary
        question_data = eval(line)
        # Add the points for the current question to the total
        total_points += question_data.get("points", 0)

# Print the total points
print(f"Total points across all questions: {total_points}")