# #!/bin/bash

# # This script runs a Python script with different methods sequentially.
# # It uses 'echo' to print the status to the console, making it clear which part is running.

# echo "--- Starting script runs ---"

# # # Run the first method: cot
# # echo "Running with method: standard"
# # python run.py --method "standard"
# # echo "Finished running with method: standard."

# # echo "--- Starting script runs ---"

# # # Run the first method: cot
# # echo "Running with method: cot"
# # python run.py --method "cot"
# # echo "Finished running with method: cot."

# # echo "" # Add a blank line for better readability

# # Run the second method: questions_first
# echo "Running with method: questions_first"
# python run.py --method "questions_first"
# echo "Finished running with method: questions_first."

# echo "" # Add a blank line for better readability

# # Run the second method: questions_first
# echo "Running with method: persona"
# python run.py --method "persona"
# echo "Finished running with method: persona."

# echo "" # Add a blank line for better readability

# # Run the second method: questions_first
# echo "Running with method: structured_decomposition"
# python run.py --method "structured_decomposition"
# echo "Finished running with method: structured_decomposition."

# echo "" # Add a blank line for better readability

# # Run the second method: questions_first
# echo "Running with method: double_check_questions"
# python run.py --method "double_check_questions"
# echo "Finished running with method: double_check_questions."

# echo "" # Add a blank line for better readability

# # Run the second method: questions_first
# echo "Running with method: one_at_a_time_focus"
# python run.py --method "one_at_a_time_focus"
# echo "Finished running with method: one_at_a_time_focus."

# echo "" # Add a blank line for better readability

# echo "Running with method: one_at_a_time_plan"
# python run.py --method "one_at_a_time_plan"
# echo "Finished running with method: one_at_a_time_plan."

# echo "" # Add a blank line for better readability

# echo "Running with method: self_refine"
# python run.py --method "self_refine"
# echo "Finished running with method: self_refine."

echo "" # Add a blank line for better readability

echo "Running with method: confidence_assessment"
python run.py --method "confidence_assessment"
echo "Finished running with method: confidence_assessment."

echo "--- All script runs completed ---"
