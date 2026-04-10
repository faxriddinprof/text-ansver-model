"""
parser.py

Reads raw text files submitted for evaluation.
"""


def read_txt(file_path):
    """Read a .txt file and return its contents as a string."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

