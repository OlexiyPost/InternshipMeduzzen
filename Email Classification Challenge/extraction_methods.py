import os
import re

def get_all_text_files(path: str, extension='.txt') -> list[str]:
    """
    Extracts all paths to files in the directory.\n
    Parameters:\n
    - path: Path to the local directory with emails.
    - extension: File extension of saved emails.\n
    Returns the list of paths to the file.
    """
    files_dirs = []
    for obj in os.listdir(path):
        obj_path = os.path.join(path, obj)

        if os.path.isfile(obj_path) and obj_path.lower().endswith(extension):
            files_dirs.append(obj_path)
        
        elif os.path.isdir(obj_path):
            files_dirs.extend(get_all_text_files(obj_path, extension))

    return files_dirs

def get_class_from_path(path: str) -> str:
    """
    Extracts email class from filepath.\n
    Parameters:\n
    - path: Path to the email file.
    Returns the email class.
    """
    pattern = re.compile(r'(\d{4}_\w+)')
    match = re.search(pattern, path)
    return match.group(1)

def extract_email_text(path: str) -> str:
    """
    Extracts text between dashes from email.\n
    Parameters:\n
    - path: Path to the email file.
    Returns the email text as a file.
    """
    with open(path, 'r') as f:
        text = f.read()
    pattern = re.compile(r'-{2,}\n(.*?)\n-{2,}', re.DOTALL)
    match = re.search(pattern, text)
    email_text = ""
    if match:
        email_text = re.sub(r'[\d\n]', ' ', match.group(1))
        email_text = re.sub(r'\d+', '', email_text)
    return email_text

