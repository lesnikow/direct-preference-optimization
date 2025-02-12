import json
import os


def debug_jsonl_file(file_path):
    """
    Debug a JSONL file by reading it line by line and identifying problematic lines.

    Args:
        file_path (str): Path to the JSONL file

    Returns:
        tuple: (is_valid, list of problematic line numbers and their contents)
    """
    problems = []
    is_valid = True

    try:
        with open(file_path, "r") as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    is_valid = False
                    problems.append(
                        {"line_number": line_num, "content": line, "error": str(e)}
                    )
    except Exception as e:
        return False, [
            {"line_number": 0, "content": None, "error": f"File error: {str(e)}"}
        ]

    return is_valid, problems


def fix_common_json_issues(content):
    """
    Attempt to fix common JSON formatting issues.

    Args:
        content (str): The problematic JSON string

    Returns:
        str: Potentially fixed JSON string
    """
    # Replace single quotes with double quotes
    content = content.replace("'", '"')

    # Remove trailing commas in objects and arrays
    content = content.replace(",}", "}").replace(",]", "]")

    # Ensure boolean values are lowercase
    content = content.replace("True", "true").replace("False", "false")

    # Ensure null values are lowercase
    content = content.replace("None", "null")

    return content


if __name__ == "__main__":
    file_path = os.path.expanduser(
        "~/fast-chat/fastchat/llm_judge/data/mt_bench/model_judgment/gpt-4-turbo_pair.jsonl"
    )
    is_valid, problems = debug_jsonl_file(file_path)

    if is_valid:
        print(f"File '{file_path}' is valid.")
    else:
        print(f"File '{file_path}' is invalid. Problems:")
        for problem in problems:
            print(f"Line {problem['line_number']}: {problem['error']}")
            print(f"Content: {problem['content']}")
            print()
