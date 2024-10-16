import chromadb
import json
import time

# Initialize ChromaDB for long-term memory storage
client = chromadb.PersistentClient(path=r"C:\Users\AM ECOSYSTEMS\OneDrive\Documents\Chatbot\AIManager\Embeddings")
long_term_memory_collection = client.get_or_create_collection(
    name="LongTermMemory",
    # Use Chroma's built-in embedding function
    embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # Example model name
    )
)

# Function to store data in long-term memory
def store_in_long_term_memory(ai_response,user_query):
    try:
        # Create a unique ID for the document (e.g., using a hash of user_query and ai_response)
        doc_id = str(hash(user_query + ai_response))
        print(f"Generated Document ID: {doc_id}")

        # Convert query and response into a JSON string
        document_content = json.dumps({"query": user_query, "response": ai_response})

        # Store the document in the Chroma collection
        long_term_memory_collection.add(
            documents=[document_content],  # List of strings (JSON format)
            ids=[doc_id]  # List containing the unique document ID
        )
        print(f"Stored in long-term memory: {user_query} -> {ai_response}")
    except Exception as e:
        print(f"Error occurred while storing data in long-term memory: {e}")

# Function to retrieve relevant data from long-term memory
def retrieve_from_long_term_memory(query):
    try:
        # Query Chroma for the most relevant document based on the query
        results = long_term_memory_collection.query(
            query_texts=[query],  # Query by text using Chroma’s built-in embeddings
            n_results=2
        )

        # Print results to inspect the structure
        print("Query Results:", results)

        # Ensure results contain valid documents
        if not results.get('documents') or not results['documents'][0] or results['documents'][0][0] is None:
            print("No relevant documents found.")
            return None

        # Parse and return the response content from the retrieved document
        document_content = results['documents'][0][0]
        print(f"Retrieved Document Content: {document_content}")

        document = json.loads(document_content)  # Parse JSON string to dictionary
        return document.get('response')
    except Exception as e:
        print(f"Error occurred while retrieving data from long-term memory: {e}")
        return None

# Function to show all data present in long-term memory along with their IDs
def show_all_long_term_memory():
    try:
        all_data = long_term_memory_collection.get()
        print(json.dumps(all_data, indent=2))  # Pretty print for better visibility

        if not all_data.get('documents'):
            print("No data found in long-term memory.")
            return None

        return all_data
    except Exception as e:
        print(f"Error occurred while retrieving all data from long-term memory: {e}")
        return None
    
# Function to clear all data in long-term memory while keeping the collection intact
def clear_all_long_term_memory():
    try:
        # Retrieve all IDs present in the collection
        all_data = long_term_memory_collection.get()
        document_ids = all_data.get('ids', [])

        # Delete all documents by their IDs
        if document_ids:
            long_term_memory_collection.delete(ids=document_ids)
            print("All data has been cleared from long-term memory.")
        else:
            print("No data found to delete.")
    except Exception as e:
        print(f"Error occurred while clearing long-term memory: {e}")


# Function to remove specific data from long-term memory by ID
def remove_data_by_id(doc_id):
    try:
        # Delete the specific document using its ID
        long_term_memory_collection.delete(ids=[doc_id])
        print(f"Document with ID {doc_id} has been removed.")
    except Exception as e:
        print(f"Error occurred while removing data by ID: {e}")

# # Example usage
# store_in_long_term_memory(
#     "What are you talking about?",
#     "Hi, we are talking about how we are planning to do Uipath automation"
# )
# show_all_long_term_memory()
# remove_data_by_id("4190926347970583280")
# show_all_long_term_memory()
# time.sleep(30)
# clear_all_long_term_memory()

# response = retrieve_from_long_term_memory("Uipath Automation")
# if response:
#     print(f"Retrieved Response: {response}")
# else:
#     print("No response found.")
