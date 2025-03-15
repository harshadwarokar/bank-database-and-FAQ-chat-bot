# Bank Customer Chat Bot

This repository contains a **Bank Customer Chat Bot** built using **Streamlit**, **ChromaDB**, **PostgreSQL**, and **Groq LLM** for natural language processing. The chatbot can answer FAQs from a vector database and execute SQL queries for customer-related information.

## Features
- **FAQ Handling**: Uses ChromaDB for fast vector search to retrieve frequently asked questions.
- **Database Querying**: Generates and executes SQL queries on a PostgreSQL database.
- **LLM-Powered Queries**: Uses **Groq LLM** to rewrite and improve user queries.
- **Streamlit UI**: Provides an interactive web interface.
- **Logging & Error Handling**: Includes logging to capture errors and debugging information.

## Installation
### Prerequisites
- Python 3.8+
- PostgreSQL Database

### Clone the Repository
```bash
git clone https://github.com/harshadwarokar/bank-database-and-FAQ-chat-bot.git
cd bank-database-and-FAQ-chat-bot
```

### Install Dependencies
Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

### Set Up Environment Variables
Create a `.env` file and configure the following variables:
```env
DB_USERNAME="postgres"
DB_PASSWORD="your_password"
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="postgres"
GROQ_API_KEY="your_groq_api_key"
```

## Running the Application
Start the Streamlit app using:
```bash
streamlit run vectordbbank.py
```

## File Structure
```
├── createvectordb.py     # Creates a vector database from FAQ data
├── vectordbbank.py       # Main chatbot application
├── requirements.txt      # Required dependencies
├── .env                  # Environment variables
```

## Usage
1. Enter an FAQ question in the text box.
2. If the question exists in the FAQ database, an answer is retrieved.
3. If no FAQ match is found, the chatbot generates an SQL query and fetches results from the database.

## Contributing
Feel free to submit pull requests or report issues.

## Contact
For any inquiries, contact harshad warokar at [harshadwarokar@gmail.com].

