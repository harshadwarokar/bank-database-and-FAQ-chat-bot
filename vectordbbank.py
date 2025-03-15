

# import os
# import re
# import logging
# import streamlit as st
# import base64
# import traceback

# from dotenv import load_dotenv
# from langchain_community.chat_models import ChatOllama
# from langchain_community.utilities import SQLDatabase
# from langchain_core.prompts import ChatPromptTemplate

# import chromadb
# from sentence_transformers import SentenceTransformer

# # -----------------------------------------------------------------------------
# # Logging configuration
# # -----------------------------------------------------------------------------
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger(__name__)

# # -----------------------------------------------------------------------------
# # Load environment variables from .env file (if any)
# # -----------------------------------------------------------------------------
# load_dotenv()

# # -----------------------------------------------------------------------------
# # Streamlit Page Configuration
# # -----------------------------------------------------------------------------
# st.set_page_config(page_title="Bank Customer Chat Bot", layout="centered")

# # -----------------------------------------------------------------------------
# # Utility Function: Convert Image to Base64 for Logo Display
# # -----------------------------------------------------------------------------
# def image_to_base64(image_path):
#     """
#     Converts an image file to a Base64 string.
#     :param image_path: Path to the image file.
#     :return: Base64 encoded string.
#     """
#     try:
#         with open(image_path, "rb") as img_file:
#             encoded = base64.b64encode(img_file.read()).decode("utf-8")
#         return encoded
#     except Exception as e:
#         logger.error(f"Error converting image {image_path} to Base64: {e}")
#         return ""

# # -----------------------------------------------------------------------------
# # SQL Database Connection and Query Execution Functions
# # -----------------------------------------------------------------------------
# def connect_database():
#     """
#     Connect to the PostgreSQL database and initialize the SQLDatabase object.
#     Database credentials can be set via environment variables or default values.
#     :return: SQLDatabase object or None if connection fails.
#     """
#     username = os.getenv("DB_USERNAME", "postgres")
#     port = os.getenv("DB_PORT", "5432")
#     host = os.getenv("DB_HOST", "localhost")
#     password = os.getenv("DB_PASSWORD", "rajyug")
#     database = os.getenv("DB_NAME", "postgres")
#     postgres_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
#     try:
#         db = SQLDatabase.from_uri(postgres_uri, sample_rows_in_table_info=3)
#         logger.info("Database connected successfully.")
#         return db
#     except Exception as e:
#         logger.error(f"Failed to connect to database: {e}")
#         st.error("Database connection failed. Please try again later.")
#         return None

# def run_sql_query(db, query):
#     """
#     Execute the provided SQL query against the database.
#     :param db: SQLDatabase instance.
#     :param query: SQL query string.
#     :return: Query result or error message.
#     """
#     try:
#         result = db.run(query)
#         if not result:
#             return "No results found."
#         return result
#     except Exception as e:
#         logger.error(f"Error executing query: {e}")
#         return f"Error executing query: {e}"

# def validate_query(query):
#     """
#     Validate SQL query to ensure it does not contain unsafe operations.
#     :param query: SQL query string.
#     :return: True if safe, False otherwise.
#     """
#     unsafe_keywords = ["insert", "update", "delete", "drop", "alter"]
#     for keyword in unsafe_keywords:
#         if keyword in query.lower():
#             logger.warning(f"Blocked query due to unsafe keyword: {keyword}")
#             return False
#     return True

# def extract_sql(query_text):
#     """
#     Extract a valid SQL query from the generated text.
#     :param query_text: Text that may contain an SQL query.
#     :return: Valid SQL query string or an empty string if not found.
#     """
#     query_text = query_text.strip()
#     if query_text.lower().startswith("select") or query_text.lower().startswith("with"):
#         return query_text
#     match = re.search(r"(SELECT\s+.*?;)", query_text, re.IGNORECASE | re.DOTALL)
#     if match:
#         extracted = match.group(1).strip()
#         logger.info(f"Extracted SQL: {extracted}")
#         return extracted
#     return ""

# # -----------------------------------------------------------------------------
# # FAQ Agent: Uses ChromaDB to Retrieve FAQs via Vector Search
# # -----------------------------------------------------------------------------
# class FAQAgent:
#     """
#     Agent to handle FAQ retrieval using ChromaDB.
#     This agent rewrites the user question (using a local LLM) to a canonical form
#     and then performs a vector search in the already existing 'bank_faqs' collection.
#     """
#     def __init__(self, chroma_collection=None):
#         # Load the embedding model for semantic search.
#         self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

#         # Connect to the existing vector database folder "faq_vd"
#         self.client = chromadb.PersistentClient(path="faqs_db")

#         # Initialize a local LLM for question rewriting.
#         try:
#             self.llm_rewriter = ChatOllama(model="llama3.1", temperature=0.1)
#         except Exception as e:
#             logger.error(f"Error initializing LLM rewriter: {e}")
#             self.llm_rewriter = None

#         # Retrieve the existing FAQ collection.
#         if chroma_collection:
#             self.collection = chroma_collection
#         else:
#             try:
#                 # Only extract the collection; do not create a new one.
#                 self.collection = self.client.get_collection(name="bank_faqs")
#             except Exception as e:
#                 logger.error(f"Error retrieving collection 'bank_faqs' from the vector database: {e}")
#                 self.collection = None

#     def rewrite_question(self, user_question):
#         """
#         Use the local LLM to rewrite the user question into a canonical form,
#         making it easier to match against the stored FAQs.
#         E.g., "Should I come on Sunday?" -> "Are you open on Sunday?"
#         """
#         if not self.llm_rewriter:
#             return user_question

#         rewrite_prompt = (
#             "Rewrite the following user question into a standard, canonical form "
#             "so it can match known FAQs. Only rewrite for clarity, don't add extra info.\n\n"
#             "User Question: {user_question}\n"
#             "Rewritten Question:"
#         )
#         try:
#             prompt = ChatPromptTemplate.from_template(rewrite_prompt)
#             chain = prompt | self.llm_rewriter
#             response = chain.invoke({
#                 "user_question": user_question
#             })
#             rewritten = response.content.strip()
#             logger.info(f"Rewritten question: {rewritten}")
#             return rewritten
#         except Exception as e:
#             logger.error(f"Error rewriting question: {e}")
#             return user_question

#     def retrieve_faq_answer(self, user_question, top_k=3, threshold=0.9):
#         """
#         Retrieve the most relevant FAQ answer using vector search.
#         The method rewrites the user question, embeds it, and then queries the existing
#         'bank_faqs' collection in the vector database.
        
#         :param user_question: User's query.
#         :param top_k: Number of top results to retrieve.
#         :param threshold: Similarity threshold for a match (lower distance is better).
#         :return: FAQ answer if found and distance < threshold, else None.
#         """
#         if not self.collection:
#             logger.error("FAQ collection not available in the vector database.")
#             return None

#         try:
#             # Rewrite user question for better matching.
#             canonical_q = self.rewrite_question(user_question)

#             # Encode the canonical question.
#             query_embedding = self.embedding_model.encode(canonical_q).tolist()

#             # Perform vector search in the pre-existing collection.
#             results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

#             # Check if a relevant FAQ answer is found.
#             if results["metadatas"] and results["distances"]:
#                 distance = results["distances"][0][0]  # distance of the top result
#                 if distance < threshold:
#                     answer = results["metadatas"][0][0].get("answer", None)
#                     logger.info(f"FAQ match found with distance {distance}: {answer}")
#                     return answer
#             return None
#         except Exception as e:
#             logger.error(f"Error retrieving FAQ answer: {e}")
#             return None


# # -----------------------------------------------------------------------------
# # SQL Agent: Uses a Local LLM (llama3.1) to Generate and Execute SQL Queries
# # -----------------------------------------------------------------------------
# class SQLAgent:
#     """
#     Agent to handle SQL queries for customer data.
#     Uses a local LLM model (llama3.1) via ChatOllama to generate SQL queries.
#     """
#     def __init__(self, db):
#         self.db = db
#         try:
#             self.llm_model = ChatOllama(model="llama3.1", temperature=0.1)
#         except Exception as e:
#             logger.error(f"Error initializing LLM model: {e}")
#             self.llm_model = None
#         self.schema = self.get_enhanced_schema()

#     def get_enhanced_schema(self):
#         """
#         Retrieve enhanced schema details from the database.
#         """
#         try:
#             if self.db:
#                 return self.db.get_table_info()
#             else:
#                 return "Database connection not available."
#         except Exception as e:
#             logger.error(f"Error retrieving schema: {e}")
#             return "Error retrieving schema."

#     def generate_query(self, question, history=""):
#         """
#         Generate an SQL query from a user question using the local LLM.
#         """
#         prompt_text = (
#             "You are a SQL expert for a banking database. Follow these rules STRICTLY:\n"
#             "1. Use ONLY these tables/columns: {schema}\n"
#             "2. Do NOT include any greetings or commentary; output only a valid SQL query ending with a semicolon.\n"
#             "3. Never invent tables/columns that don't exist.\n"
#             "4. If you are unsure, output exactly: I don't know\n"
#             "5. Use PostgreSQL syntax only.\n"
#             "6. Chat History: {history}\n"
#             "Question: {question}\n"
#             "SQL Query:"
#         )
#         try:
#             if not self.llm_model:
#                 return None

#             prompt = ChatPromptTemplate.from_template(prompt_text)
#             chain = prompt | self.llm_model
#             response = chain.invoke({
#                 "question": question,
#                 "schema": self.schema,
#                 "history": history
#             })
#             raw_query = response.content.strip()
#             sql_query = extract_sql(raw_query)
#             if not sql_query:
#                 raise Exception("Generated output does not contain a valid SQL query.")
#             logger.info(f"Generated SQL query: {sql_query}")
#             return sql_query
#         except Exception as e:
#             logger.error(f"Error generating SQL query: {e}")
#             return None

#     def execute_query(self, sql_query):
#         """
#         Execute the given SQL query on the database.
#         """
#         try:
#             if not validate_query(sql_query):
#                 return "Query blocked: Contains unsafe operations."
#             return run_sql_query(self.db, sql_query)
#         except Exception as e:
#             logger.error(f"Error executing SQL query: {e}")
#             return f"Error executing SQL query: {e}"

#     def summarize_result(self, raw_result):
#         """
#         Summarize the raw result with NO information lost, but in a more readable format.
#         """
#         if isinstance(raw_result, list):
#             rows_str = "\n".join(str(row) for row in raw_result)
#             formatted_result = rows_str
#         else:
#             formatted_result = str(raw_result)

#         if not self.llm_model:
#             # If summarizing LLM is unavailable, return the raw data
#             return formatted_result

#         prompt_text = (
#             "think of yourself as an bank chat bot assistance "
#             "The following is the raw output from an SQL query. Reformat it in a user-friendly way "
#             "without losing any information. Keep all data, but present it clearly.\n\n"
#             "Raw Data:\n{formatted_result}\n\n"
#             "Reformatted (No Info Lost):"
#         )

#         try:
#             prompt = ChatPromptTemplate.from_template(prompt_text)
#             chain = prompt | self.llm_model
#             response = chain.invoke({
#                 "formatted_result": formatted_result
#             })
#             return response.content.strip()
#         except Exception as e:
#             logger.error(f"Error summarizing result: {e}")
#             return formatted_result

#     def process_question(self, question, history=""):
#         """
#         Process a customer query by generating an SQL query, executing it, and returning
#         BOTH the raw query results and a no-information-lost summary.
#         """
#         try:
#             sql_query = self.generate_query(question, history)
#             if not sql_query:
#                 return "I'm sorry, I couldn't generate a valid SQL query."

#             result = self.execute_query(sql_query)
#             summary = self.summarize_result(result)

#             return f"**Raw Results:**\n{result}\n\n**Summary (No Info Lost):**\n{summary}"

#         except Exception as e:
#             logger.error(f"Error processing SQL question: {e}")
#             return "We encountered an error processing your SQL query. Please try again later."

# # -----------------------------------------------------------------------------
# # Combined Bank ChatBot Agent: Integrates FAQAgent and SQLAgent
# # -----------------------------------------------------------------------------
# class BankChatBot:
#     """
#     Combined chatbot that first checks the FAQAgent (ChromaDB) for a relevant answer.
#     If no FAQ answer is found, it falls back to the SQLAgent to handle customer data queries.
#     """
#     def __init__(self, db):
#         self.db = db
#         self.faq_agent = FAQAgent()
#         self.sql_agent = SQLAgent(db)
#         self.chat_history = []  # Internal chat history as a list of messages

#     def update_history(self, role, content):
#         """
#         Update internal chat history.
#         :param role: "User" or "Assistant".
#         :param content: The message content.
#         """
#         self.chat_history.append({"role": role, "content": content})

#     def process_question(self, question):
#         """
#         Process user question by first checking FAQAgent, then SQLAgent if needed.
#         :param question: The user's question.
#         :return: Answer string.
#         """
#         try:
#             self.update_history("User", question)

#             # 1) Check FAQ
#             faq_answer = self.faq_agent.retrieve_faq_answer(question)
#             if faq_answer:
#                 self.update_history("Assistant", faq_answer)
#                 return faq_answer

#             # 2) If no FAQ match, try SQL
#             sql_answer = self.sql_agent.process_question(question)
#             self.update_history("Assistant", sql_answer)
#             return sql_answer
#         except Exception as e:
#             logger.error(f"Error processing combined question: {e}")
#             return "We encountered an error processing your question. Please try again later."

#     def get_chat_history(self):
#         """
#         Return the chat history as a formatted string.
#         """
#         history_str = ""
#         for msg in self.chat_history:
#             role = msg.get("role", "unknown")
#             content = msg.get("content", "")
#             history_str += f"{role}: {content}\n"
#         return history_str

# # -----------------------------------------------------------------------------
# # Streamlit User Interface (Text-to-Text Only)
# # -----------------------------------------------------------------------------
# def main():
#     """
#     Main function to run the Streamlit app.
#     """
#     st.title("Bank Customer Chat Bot")
    
#     # Display header logos (optional; update paths as needed).
#     LOGO_PATH = "logo.jpg"
#     USER_LOGO_PATH = "user.jpg"
#     logo_base64 = image_to_base64(LOGO_PATH)
#     user_logo_base64 = image_to_base64(USER_LOGO_PATH)
#     st.markdown(
#         f"""
#         <div style="display:flex; justify-content:space-between; align-items:center;">
#             <img src="data:image/jpeg;base64,{logo_base64}" alt="Bank Logo" style="height:80px;">
#             <img src="data:image/jpeg;base64,{user_logo_base64}" alt="User Logo" style="height:80px; border-radius:50%;">
#         </div>
#         """,
#         unsafe_allow_html=True
#     )
    
#     # Connect to the SQL database.
#     db = connect_database()
#     if not db:
#         st.error("Unable to connect to the database.")
#         return

#     # Initialize the combined BankChatBot agent and store in session state.
#     if "bank_chatbot" not in st.session_state:
#         st.session_state.bank_chatbot = BankChatBot(db)

#     # Text input for user question.
#     user_question = st.text_input("Enter your question:", "")
#     if st.button("Submit") and user_question:
#         try:
#             response = st.session_state.bank_chatbot.process_question(user_question)
#             st.write(f"**Assistant:** {response}")
#         except Exception as e:
#             st.error("We encountered an error processing your question.")
#             logger.error(f"Error in main: {e}\n{traceback.format_exc()}")

#     # Option to display chat history.
#     if st.checkbox("Show Chat History"):
#         st.text_area("Chat History", st.session_state.bank_chatbot.get_chat_history(), height=300)

#     # Footer
#     st.markdown(
#         """
#         <div style="text-align:center; margin-top:20px;">
#             <p>Developed by @Rajyug It Solutions Pvt. Ltd</p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# # -----------------------------------------------------------------------------
# # Main Execution Block
# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         st.error("An error occurred in the application.")
#         logger.error(f"Unhandled exception: {e}\n{traceback.format_exc()}")

import os
import re
import logging
import streamlit as st
import base64
import traceback

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate

import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Load environment variables from .env file (if any)
# -----------------------------------------------------------------------------
load_dotenv()

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Bank Customer Chat Bot", layout="centered")

# -----------------------------------------------------------------------------
# Utility Function: Convert Image to Base64 for Logo Display
# -----------------------------------------------------------------------------
def image_to_base64(image_path):
    """
    Converts an image file to a Base64 string.
    :param image_path: Path to the image file.
    :return: Base64 encoded string.
    """
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
        return encoded
    except Exception as e:
        logger.error(f"Error converting image {image_path} to Base64: {e}")
        return ""

# -----------------------------------------------------------------------------
# SQL Database Connection and Query Execution Functions
# -----------------------------------------------------------------------------
def connect_database():
    """
    Connect to the PostgreSQL database and initialize the SQLDatabase object.
    Database credentials can be set via environment variables or default values.
    :return: SQLDatabase object or None if connection fails.
    """
    username = os.getenv("DB_USERNAME", "postgres")
    port = os.getenv("DB_PORT", "5432")
    host = os.getenv("DB_HOST", "localhost")
    password = os.getenv("DB_PASSWORD", "rajyug")
    database = os.getenv("DB_NAME", "postgres")
    postgres_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
    try:
        db = SQLDatabase.from_uri(postgres_uri, sample_rows_in_table_info=3)
        logger.info("Database connected successfully.")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        st.error("Database connection failed. Please try again later.")
        return None

def run_sql_query(db, query):
    """
    Execute the provided SQL query against the database.
    :param db: SQLDatabase instance.
    :param query: SQL query string.
    :return: Query result or error message.
    """
    try:
        result = db.run(query)
        if not result:
            return "No results found."
        return result
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return f"Error executing query: {e}"

def validate_query(query):
    """
    Validate SQL query to ensure it does not contain unsafe operations.
    :param query: SQL query string.
    :return: True if safe, False otherwise.
    """
    unsafe_keywords = ["insert", "update", "delete", "drop", "alter"]
    for keyword in unsafe_keywords:
        if keyword in query.lower():
            logger.warning(f"Blocked query due to unsafe keyword: {keyword}")
            return False
    return True

def extract_sql(query_text):
    """
    Extract a valid SQL query from the generated text.
    :param query_text: Text that may contain an SQL query.
    :return: Valid SQL query string or an empty string if not found.
    """
    query_text = query_text.strip()
    if query_text.lower().startswith("select") or query_text.lower().startswith("with"):
        return query_text
    match = re.search(r"(SELECT+.*?;)", query_text, re.IGNORECASE | re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        logger.info(f"Extracted SQL: {extracted}")
        return extracted
    return ""

# -----------------------------------------------------------------------------
# FAQ Agent: Uses ChromaDB to Retrieve FAQs via Vector Search
# -----------------------------------------------------------------------------
class FAQAgent:
    """
    Agent to handle FAQ retrieval using ChromaDB.
    This agent rewrites the user question (using a local LLM) to a canonical form
    and then performs a vector search in the already existing 'bank_faqs' collection.
    """
    def __init__(self, chroma_collection=None):
        # Load the embedding model for semantic search.
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Connect to the existing vector database folder "faq_vd"
        self.client = chromadb.PersistentClient(path="faqvectordatabase")

        # Initialize a Groq LLM for question rewriting.
        try:
            self.llm_rewriter = ChatGroq(
                model="mixtral-8x7b-32768",
                temperature=0.1
            )
        except Exception as e:
            logger.error(f"Error initializing LLM rewriter: {e}")
            self.llm_rewriter = None

        # Retrieve the existing FAQ collection.
        if chroma_collection:
            self.collection = chroma_collection
        else:
            try:
                # Only extract the collection; do not create a new one.
                self.collection = self.client.get_collection(name="faqvectordatabase")
            except Exception as e:
                logger.error(f"Error retrieving collection 'bank_faqs' from the vector database: {e}")
                self.collection = None

    def rewrite_question(self, user_question):
        """
        Use the local LLM to rewrite the user question into a canonical form,
        making it easier to match against the stored FAQs.
        E.g., "Should I come on Sunday?" -> "Are you open on Sunday?"
        """
        if not self.llm_rewriter:
            return user_question

        rewrite_prompt = (
            "Rewrite the following user question into a standard, canonical form "
            "so it can match known FAQs. Only rewrite for clarity, don't add extra info.\n\n"
            "User Question: {user_question}\n"
            "Rewritten Question:"
        )
        try:
            prompt = ChatPromptTemplate.from_template(rewrite_prompt)
            chain = prompt | self.llm_rewriter
            response = chain.invoke({
                "user_question": user_question
            })
            rewritten = response.content.strip()
            logger.info(f"Rewritten question: {rewritten}")
            return rewritten
        except Exception as e:
            logger.error(f"Error rewriting question: {e}")
            return user_question

    def retrieve_faq_answer(self, user_question, top_k=3, threshold=0.9):
        """
        Retrieve the most relevant FAQ answer using vector search.
        The method rewrites the user question, embeds it, and then queries the existing
        'bank_faqs' collection in the vector database.
        
        :param user_question: User's query.
        :param top_k: Number of top results to retrieve.
        :param threshold: Similarity threshold for a match (lower distance is better).
        :return: FAQ answer if found and distance < threshold, else None.
        """
        if not self.collection:
            logger.error("FAQ collection not available in the vector database.")
            return None

        try:
            # Rewrite user question for better matching.
            canonical_q = self.rewrite_question(user_question)

            # Encode the canonical question.
            query_embedding = self.embedding_model.encode(canonical_q).tolist()

            # Perform vector search in the pre-existing collection.
            results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

            # Check if a relevant FAQ answer is found.
            if results["metadatas"] and results["distances"]:
                distance = results["distances"][0][0]  # distance of the top result
                if distance < threshold:
                    answer = results["metadatas"][0][0].get("answer", None)
                    logger.info(f"FAQ match found with distance {distance}: {answer}")
                    return answer
            return None
        except Exception as e:
            logger.error(f"Error retrieving FAQ answer: {e}")
            return None

# -----------------------------------------------------------------------------
# SQL Agent: Uses a Local LLM (Groq) to Generate and Execute SQL Queries
# -----------------------------------------------------------------------------
class SQLAgent:
    """
    Agent to handle SQL queries for customer data.
    Uses a local LLM model (Groq) via ChatGroq to generate SQL queries.
    """
    def __init__(self, db):
        self.db = db
        try:
            self.llm_model = ChatGroq(
                model="mixtral-8x7b-32768",
                temperature=0.1
            )
        except Exception as e:
            logger.error(f"Error initializing LLM model: {e}")
            self.llm_model = None
        self.schema = self.get_enhanced_schema()

    def get_enhanced_schema(self):
        """
        Retrieve enhanced schema details from the database.
        """
        try:
            if self.db:
                return self.db.get_table_info()
            else:
                return "Database connection not available."
        except Exception as e:
            logger.error(f"Error retrieving schema: {e}")
            return "Error retrieving schema."

    def generate_query(self, question, history=""):
        """
        Generate an SQL query from a user question using the local LLM.
        """
        prompt_text = (
            "You are a SQL expert for a banking database. Follow these rules STRICTLY:\n"
            "1. Use ONLY these tables/columns: {schema}\n"
            "2. Do NOT include any greetings or commentary; output only a valid SQL query ending with a semicolon.\n"
            "3. Never invent tables/columns that don't exist.\n"
            "4. If you are unsure, output exactly: I don't know\n"
            "5. Use PostgreSQL syntax only.\n"
            "6. Chat History: {history}\n"
            "Question: {question}\n"
            "SQL Query:"
        )
        try:
            if not self.llm_model:
                return None

            prompt = ChatPromptTemplate.from_template(prompt_text)
            chain = prompt | self.llm_model
            response = chain.invoke({
                "question": question,
                "schema": self.schema,
                "history": history
            })
            raw_query = response.content.strip()
            sql_query = extract_sql(raw_query)
            if not sql_query:
                raise Exception("Generated output does not contain a valid SQL query.")
            logger.info(f"Generated SQL query: {sql_query}")
            return sql_query
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            return None

    def execute_query(self, sql_query):
        """
        Execute the given SQL query on the database.
        """
        try:
            if not validate_query(sql_query):
                return "Query blocked: Contains unsafe operations."
            return run_sql_query(self.db, sql_query)
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            return f"Error executing SQL query: {e}"

    def summarize_result(self, raw_result):
        """
        Summarize the raw result with NO information lost, but in a more readable format.
        """
        if isinstance(raw_result, list):
            rows_str = "\n".join(str(row) for row in raw_result)
            formatted_result = rows_str
        else:
            formatted_result = str(raw_result)

        if not self.llm_model:
            # If summarizing LLM is unavailable, return the raw data
            return formatted_result

        prompt_text = (
            "think of yourself as an bank chat bot assistance "
            "The following is the raw output from an SQL query. Reformat it in a user-friendly way "
            "without losing any information. Keep all data, but present it clearly.\n\n"
            "Raw Data:\n{formatted_result}\n\n"
            "Reformatted (No Info Lost):"
        )

        try:
            prompt = ChatPromptTemplate.from_template(prompt_text)
            chain = prompt | self.llm_model
            response = chain.invoke({
                "formatted_result": formatted_result
            })
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error summarizing result: {e}")
            return formatted_result

    def process_question(self, question, history=""):
        """
        Process a customer query by generating an SQL query, executing it, and returning
        BOTH the raw query results and a no-information-lost summary.
        """
        try:
            sql_query = self.generate_query(question, history)
            if not sql_query:
                return "I'm sorry, I couldn't generate a valid SQL query."

            result = self.execute_query(sql_query)
            summary = self.summarize_result(result)

            return f"**Raw Results:**\n{result}\n\n**Summary (No Info Lost):**\n{summary}"
        except Exception as e:
            logger.error(f"Error processing SQL question: {e}")
            return "We encountered an error processing your SQL query. Please try again later."

# -----------------------------------------------------------------------------
# Combined Bank ChatBot Agent: Integrates FAQAgent and SQLAgent
# -----------------------------------------------------------------------------
class BankChatBot:
    """
    Combined chatbot that first checks the FAQAgent (ChromaDB) for a relevant answer.
    If no FAQ answer is found, it falls back to the SQLAgent to handle customer data queries.
    """
    def __init__(self, db):
        self.db = db
        self.faq_agent = FAQAgent()
        self.sql_agent = SQLAgent(db)
        self.chat_history = []  # Internal chat history as a list of messages

    def update_history(self, role, content):
        """
        Update internal chat history.
        :param role: "User" or "Assistant".
        :param content: The message content.
        """
        self.chat_history.append({"role": role, "content": content})

    def process_question(self, question):
        """
        Process user question by first checking FAQAgent, then SQLAgent if needed.
        :param question: The user's question.
        :return: Answer string.
        """
        try:
            self.update_history("User", question)

            # 1) Check FAQ
            faq_answer = self.faq_agent.retrieve_faq_answer(question)
            if faq_answer:
                self.update_history("Assistant", faq_answer)
                return faq_answer

            # 2) If no FAQ match, try SQL
            sql_answer = self.sql_agent.process_question(question)
            self.update_history("Assistant", sql_answer)
            return sql_answer
        except Exception as e:
            logger.error(f"Error processing combined question: {e}")
            return "We encountered an error processing your question. Please try again later."

    def get_chat_history(self):
        """
        Return the chat history as a formatted string.
        """
        history_str = ""
        for msg in self.chat_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
        return history_str

# -----------------------------------------------------------------------------
# Streamlit User Interface (Text-to-Text Only)
# -----------------------------------------------------------------------------
def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Bank Customer Chat Bot")
    
    # Display header logos (optional; update paths as needed).
    LOGO_PATH = "logo.jpg"
    USER_LOGO_PATH = "user.jpg"
    logo_base64 = image_to_base64(LOGO_PATH)
    user_logo_base64 = image_to_base64(USER_LOGO_PATH)
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <img src="data:image/jpeg;base64,{logo_base64}" alt="Bank Logo" style="height:80px;">
            <img src="data:image/jpeg;base64,{user_logo_base64}" alt="User Logo" style="height:80px; border-radius:50%;">
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Connect to the SQL database.
    db = connect_database()
    if not db:
        st.error("Unable to connect to the database.")
        return

    # Initialize the combined BankChatBot agent and store in session state.
    if "bank_chatbot" not in st.session_state:
        st.session_state.bank_chatbot = BankChatBot(db)

    # Text input for user question.
    user_question = st.text_input("Enter your question:", "")
    if st.button("Submit") and user_question:
        try:
            response = st.session_state.bank_chatbot.process_question(user_question)
            st.write(f"**Assistant:** {response}")
        except Exception as e:
            st.error("We encountered an error processing your question.")
            logger.error(f"Error in main: {e}\n{traceback.format_exc()}")

    # Option to display chat history.
    if st.checkbox("Show Chat History"):
        st.text_area("Chat History", st.session_state.bank_chatbot.get_chat_history(), height=300)

    # Footer
    st.markdown(
        """
        <div style="text-align:center; margin-top:20px;">
            <p>Developed by @Rajyug It Solutions Pvt. Ltd</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# Main Execution Block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An error occurred in the application.")
        logger.error(f"Unhandled exception: {e}\n{traceback.format_exc()}")
