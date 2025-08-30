from rag.document_loader import ScriptLoader
from rag.rag_pipeline import create_rag_chain
from rag.utils import select_lesson_file
from integration.input import get_transcript
print(get_transcript())
def main():
    SCRIPTS_PATH = 'data/scripts/shrek'

    print("Please select a lesson plan file to begin...")
    lesson_file_path = select_lesson_file()

    if not lesson_file_path:
        print("No lesson file selected. Exiting the program.")
        return

    print(f"Loaded lesson: {lesson_file_path}")
    # Here, you would add your lesson parsing logic from the previous step
    # lesson = parse_lesson(lesson_file_path)

    # Choose character
    character_name = input("Which character would you like to talk to? (e.g., Shrek, Donkey): ")

    # Load documents for the specified character
    print(f"Loading dialogue for {character_name}...")
    loader = ScriptLoader(directory_path=SCRIPTS_PATH, character_name=character_name)
    docs = loader.load()

    # Check if any documents were loaded
    if not docs:
        print(f"Sorry, I couldn't find any dialogue for '{character_name}'. Please check the name and try again.")
        return

    # rag chain
    rag_chain = create_rag_chain(docs, character_name)

    print(f"\n{character_name.capitalize()} is ready to talk! (Type 'exit' to quit)")

    # Start a conversation loop
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'exit':
            print(f"{character_name.capitalize()}: 'Bye-bye!'")
            break

        # Invoke the chain and print the answer
        answer = rag_chain.invoke(question)
        print(f"{character_name.capitalize()} says: {answer}")


if __name__ == "__main__":
    main()