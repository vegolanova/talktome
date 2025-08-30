from document_loader import ScriptLoader
from rag_pipeline import create_rag_chain, create_tutor_chain
from utils import select_lesson_file
from langchain_mistralai.chat_models import ChatMistralAI
from lesson_parser import parse_lesson


def main():
    SCRIPTS_PATH = '../data/scripts/shrek'

    # --- 1. Load Lesson ---
    print("Please select a lesson plan file to begin...")
    lesson_file_path = select_lesson_file()
    if not lesson_file_path:
        print("No lesson file selected. Exiting.")
        return

    lesson = parse_lesson(lesson_file_path)
    if not lesson["questions"]:
        print("Could not find any questions in the lesson plan. Exiting.")
        return

    # --- 2. Setup Character ---
    character_name = input("Which character will be the tutor? (e.g., Shrek): ")
    loader = ScriptLoader(directory_path=SCRIPTS_PATH, character_name=character_name)
    docs = loader.load()
    if not docs:
        print(f"Sorry, couldn't find dialogue for '{character_name}'.")
        return

    # --- 3. Create Chains ---
    # We need a base RAG chain for the character to introduce things
    # and a more complex Tutor chain to evaluate answers.
    base_rag_chain = create_rag_chain(docs, character_name)

    # We create the retriever and LLM once to pass into our tutor chain
    # This is a simplified version of what you'd do in a real app
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0.7)

    tutor_chain, memory = create_tutor_chain(llm, retriever, character_name, lesson["instructions"])

    # --- 4. Start Lesson ---
    intro = base_rag_chain.invoke(f"Introduce yourself and say you're about to start a lesson on multiplication.")
    print(f"\n{character_name.capitalize()}: {intro}")

    state = {"current_question": 1, "attempts": 0, "hints_used": 0}

    while state["current_question"] <= len(lesson["questions"]):
        q_data = lesson["questions"][state["current_question"]]

        # Have the character ask the question
        question_prompt = f"In your own words, ask me this question: {q_data['q']}"
        formatted_question = base_rag_chain.invoke(question_prompt)
        user_answer = input(f"\n{character_name.capitalize()}: {formatted_question}\nYour answer: ")

        # Use the tutor chain to evaluate the answer
        evaluation = tutor_chain.invoke(user_answer)
        print(f"{character_name.capitalize()}: {evaluation}")

        # Simple logic to advance the quiz
        if "CORRECT" in evaluation:
            state["current_question"] += 1
            state["attempts"] = 0
            state["hints_used"] = 0
            if state["current_question"] > len(lesson["questions"]):
                print("\n--- Lesson Complete! ---")
        else:
            state["attempts"] += 1
            if "HINT" in evaluation:
                state["hints_used"] += 1


if __name__ == "__main__":
    main()