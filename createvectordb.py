import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import ast
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    st.title("FAQ Vector Database Creator")
    st.write(
        "Enter your FAQ data in the following format (each FAQ separated by a comma):\n\n"
        '```\n'
        '"What are the bank working hours?": (\n'
        '    "Our bank operates from 9 AM to 5 PM, Monday to Friday. "\n'
        '    "We are closed on weekends (Saturday & Sunday)."\n'
        '),\n'
        '"Are you open on Sunday?": (\n'
        '    "No, we are closed on weekends (Saturday & Sunday)."\n'
        '),\n'
        '```\n\n'
        "Note: Do not include outer curly braces."
    )

    faq_text = st.text_area("Enter FAQ data:", height=300)

    if st.button("Create Vector Database"):
        if not faq_text.strip():
            st.error("Please provide FAQ data in the specified format.")
            return

        try:
            # Wrap the text with curly braces to form a valid Python dictionary
            faq_text_wrapped = "{" + faq_text + "}"
            faq_data = ast.literal_eval(faq_text_wrapped)
            if not isinstance(faq_data, dict):
                st.error("Parsed FAQ data is not a dictionary. Please check your input.")
                return
        except Exception as e:
            st.error("Error parsing FAQ data. Please ensure it is in the correct Python dictionary format.")
            logger.error(f"Parsing error: {e}")
            return

        st.write("### FAQ Data to be added:")
        st.json(faq_data)

        try:
            # Initialize the embedding model
            model = SentenceTransformer("all-MiniLM-L6-v2")

            # Connect to or create a persistent ChromaDB vector database in "faqvectordatabase"
            client = chromadb.PersistentClient(path="faqvectordatabase")

            # Retrieve or create the collection named "bank_faqs"
            collection = client.get_or_create_collection(name="bank_faqs")

            added_count = 0
            for question, answer in faq_data.items():
                try:
                    # Check if the FAQ already exists by its question (used as the unique id)
                    existing = collection.get(ids=[question])
                    if existing and len(existing["ids"]) > 0:
                        logger.info(f"FAQ already exists: {question}")
                        continue
                except Exception as ex:
                    logger.info(f"FAQ {question} not found. Proceeding to add.")

                # Generate the embedding for the question
                embedding = model.encode(question).tolist()

                # Add the FAQ to the vector database collection
                collection.add(
                    documents=[question],
                    metadatas=[{"answer": answer}],
                    ids=[question]
                )
                added_count += 1

            st.success(f"Vector database updated successfully. {added_count} FAQs added.")
        except Exception as e:
            st.error(f"Error creating/updating vector database: {e}")
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
