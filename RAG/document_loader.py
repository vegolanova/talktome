import os
from typing import List
from langchain.docstore.document import Document
from langchain_community.document_loaders.base import BaseLoader

class ScriptLoader(BaseLoader):
    def __init__(self, directory_path: str, character_name: str):
        self.directory_path = directory_path
        self.character_name = character_name.capitalize()

    def load(self) -> List[Document]:
        docs = []
        character_line_prefix = f"{self.character_name}:"

        for filename in os.listdir(self.directory_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.directory_path, filename)
                character_lines = []

                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip().startswith(character_line_prefix):
                            dialogue = line.strip().replace(character_line_prefix, "").strip()
                            character_lines.append(dialogue)

                # Only create a document if we found lines for the character in this file
                if character_lines:
                    full_text = " ".join(character_lines)
                    docs.append(Document(
                        page_content=full_text,
                        metadata={"source": filename, "character": self.character_name}
                    ))
        return docs