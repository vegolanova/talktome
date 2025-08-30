import tkinter as tk
from tkinter import filedialog

def select_lesson_file():
    root = tk.Tk()
    root.withdraw()

    # Open the file dialog, restricted to .txt files
    file_path = filedialog.askopenfilename(
        title="Select a Lesson Plan File",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )

    return file_path