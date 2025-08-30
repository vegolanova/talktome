import re


def parse_lesson(file_path):
    """
    Parses a lesson plan file into a structured dictionary.
    """
    lesson = {"instructions": "", "questions": {}}
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    instructions_match = re.search(r"\[INSTRUCTIONS\]\n(.*?)\n\[QUESTIONS\]", content, re.DOTALL)
    if instructions_match:
        lesson["instructions"] = instructions_match.group(1).strip()

    questions_section = content.split("[QUESTIONS]")[1]

    # Regex to find Q, H, and A blocks for each question number
    question_blocks = re.findall(r"Q(\d+):(.*?)\nH\d+:(.*?)\nA\d+:(.*?)(?=\nQ|\Z)", questions_section, re.DOTALL)

    for block in question_blocks:
        q_num = int(block[0])
        lesson["questions"][q_num] = {
            "q": block[1].strip(),
            "h": block[2].strip(),
            "a": block[3].strip()
        }

    return lesson