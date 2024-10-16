from get_api import get_api_key_from_json
import os
from langchain_groq import ChatGroq
from ReadFile import read_file
import asyncio
from collections import deque
from GmeetHear import start_transcription
from GmeetSpeak import speak
import chromadb
from langchain_community.vectorstores import Chroma
from langchain.agents import Tool
from LongTermMemory import retrieve_from_long_term_memory, store_in_long_term_memory

# Initialize short-term memory
short_term_memory = deque(maxlen=2)

# Function to check if the user wants to end the conversation
def check_conversation_end(user_input):
    end_phrases = ["bye", "thank you", "goodbye", "see you", "thanks"]
    return any(phrase in user_input.lower() for phrase in end_phrases)

# Tool to retrieve relevant data from long-term memory
def query_long_term_memory_tool(topic: str) -> str:
    """Tool to query long-term memory for a given topic."""
    result = retrieve_from_long_term_memory(topic)
    return result if result else "No relevant data found for the topic."

def write_report_and_send_email_tool(topic: str) -> str:
    """Tool to write a report and send an email to the user."""
    asyncio.run(
                speak(
                    text="Report Sent to user Successfully " + topic, 
                    voice_number=0, rate="+7%", pitch="+10Hz"
                )
            )
    return "Report written and email sent successfully."

# Initialize tools for LLM
tools = [
    Tool(
        name="QueryLongTermMemory",
        func=query_long_term_memory_tool,
        description="Use this tool to fetch the latest update on a given topic from long-term memory."
    ),
    Tool(
        name="WriteReportandSendEmail",
        func=write_report_and_send_email_tool,
        description="Use this tool to write a report and send an email to the user."
    )
]

def Reader():
    api_key = get_api_key_from_json(
        r"C:\Users\AM ECOSYSTEMS\OneDrive\Documents\Chatbot\Retail AI Store Bot\Apikey.json", 
        "Manager_AI"
    )

    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = api_key

    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Initial greeting
    asyncio.run(
        speak(
            text="Hi, team. Good evening! Could you tell me what progress is being made on the automation project?", 
            voice_number=1, rate="+3%", pitch="+5Hz"
        )
    )

    try:
        while True:
            print("Starting recording...")
            transcript = start_transcription(
                chunk_duration=8, sample_rate=16000,
                loudness_start_threshold=0.1, loudness_stop_threshold=0.08,
                repeating_word_limit=3, silence_timeout=2,
                model_type="small.en", vac_input_device=0, channels=1
            )
            print("Recording Stopped")

            short_term_memory.append(transcript)
            print("Short-term memory:", list(short_term_memory))

            # Retrieve relevant past information
            relevant_past_info = retrieve_from_long_term_memory(transcript) or ""
            memory_context = " ".join(short_term_memory)

            # Read the prompt from the file
            query = read_file(
                r"C:\Users\AM ECOSYSTEMS\OneDrive\Documents\Chatbot\AIManager\prompt.txt"
            )

            # Prepare input text for the LLM
            input_text = (
                f"{transcript} Reformat this by removing spelling mistakes and grammatical errors. "
                "Ignore repeating words like 'i has the thankyou i has the thankyou'. "
                "If the conversation implies the user wants to end it, add 'okay bye' to the output. "
                f"Further, here is relevant chat history: {relevant_past_info}."
                f"Here is the latest update: {memory_context} Add it in the output if it is relevant."
                "Make sure you format the output as it will sound sensible and dont give any solutions just reformat the transcript said by the user."
            )

            # Invoke the LLM
            filtered_response = llm.invoke(input_text)

            # Prepare final response with memory context
            response_text = (
                f"{query} :- {filtered_response.content} "
                f"Here is a summary of the previous conversations: {memory_context}."
                f"Dont give output as a conversation , give it as a manager not a full conversation between you and the user."
                f"Make sure you talk as a manager not preamble."
            )

            # Get AI response
            ai_response = llm.invoke(response_text)
            print("AI Response:", ai_response.content)

            print("Playing Audio")
            asyncio.run(
                speak(
                    text=ai_response.content, 
                    voice_number=1, rate="+7%", pitch="+10Hz"
                )
            )

            # Store the conversation in long-term memory
            store_in_long_term_memory(
                user_query=filtered_response.content, 
                ai_response=ai_response.content
            )

            # Check if the user wants to end the conversation
            if check_conversation_end(transcript):
                asyncio.run(
                    speak(
                        text="Thank you, goodbye!", 
                        voice_number=1, rate="+3%", pitch="+5Hz"
                    )
                )
                break

    except KeyboardInterrupt:
        print("Conversation loop stopped.")
    except TypeError as e:
        print(f"TypeError: {e}")
        print("Ensure that all objects passed to the LLM are JSON serializable.")
# Example of how to call the Reader function
if __name__ == "__main__":
    Reader()


# ------------------------------------------------------------------------------------------------#
#below code is for testing the tools
# ------------------------------------------------------------------------------------------------#
# from get_api import get_api_key_from_json
# import os
# from langchain_groq import ChatGroq
# from ReadFile import read_file
# from TranscripeUserData import start_recording_and_transcribing
# import asyncio
# from collections import deque
# from GmeetHear import start_transcription
# from GmeetSpeak import speak
# import chromadb
# from langchain_community.vectorstores import Chroma
# from langchain.agents import Tool
# from langchain_ollama import OllamaEmbeddings
# from langchain import LLMChain
# from langchain.prompts import PromptTemplate
# from LongTermMemory import retrieve_from_long_term_memory, store_in_long_term_memory

# # Initialize short-term memory
# short_term_memory = deque(maxlen=2)

# # Function to check if the user wants to end the conversation
# def check_conversation_end(user_input):
#     end_phrases = ["bye", "thank you", "goodbye", "see you", "thanks"]
#     return any(phrase in user_input.lower() for phrase in end_phrases)

# # Tool to retrieve relevant data from long-term memory
# def query_long_term_memory_tool(topic: str) -> str:
#     """Tool to query long-term memory for a given topic."""
#     result = retrieve_from_long_term_memory(topic)
#     return result if result else "No relevant data found for the topic."

# def write_report_and_send_email_tool(topic: str) -> str:
#     """Tool to write a report and send an email to the user."""
#     asyncio.run(
#         speak(
#             text="Report Sent to user Successfully " + topic, 
#             voice_number=0, rate="+7%", pitch="+10Hz"
#         )
#     )
#     return "Report written and email sent successfully."

# # Initialize tools for LLM
# tools = [
#     Tool(
#         name="QueryLongTermMemory",
#         func=query_long_term_memory_tool,
#         description="Use this tool to fetch the latest update on a given topic from long-term memory."
#     ),
#     Tool(
#         name="WriteReportandSendEmail",
#         func=write_report_and_send_email_tool,
#         description="Use this tool to write a report and send an email to the user."
#     )
# ]

# async def Reader():
#     api_key = get_api_key_from_json(
#         r"C:\Users\AM ECOSYSTEMS\OneDrive\Documents\Chatbot\Retail AI Store Bot\Apikey.json", 
#         "Manager_AI"
#     )

#     if "GROQ_API_KEY" not in os.environ:
#         os.environ["GROQ_API_KEY"] = api_key

#     llm = ChatGroq(
#         model="llama-3.1-70b-versatile",
#         temperature=0,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#         model_kwargs={"tools": tools}
#     )

#     # Initial greeting
#     await speak(
#         text="Hi, team. Good evening! Could you tell me what progress is being made on the automation project?", 
#         voice_number=1, rate="+3%", pitch="+5Hz"
#     )

#     try:
#         while True:
#             print("Starting recording...")
#             transcript = start_transcription(
#                 chunk_duration=8, sample_rate=16000,
#                 loudness_start_threshold=0.1, loudness_stop_threshold=0.08,
#                 repeating_word_limit=3, silence_timeout=2,
#                 model_type="small.en", vac_input_device=0, channels=1
#             )
#             print("Recording Stopped")

#             if transcript:
#                 short_term_memory.append(transcript)
#                 print("Short-term memory:", list(short_term_memory))
#             else:
#                 print("Transcript is empty, skipping.")
#                 continue

#             relevant_past_info = retrieve_from_long_term_memory(transcript) or ""
#             memory_context = " ".join(short_term_memory)

#             try:
#                 query = read_file(
#                     r"C:\Users\AM ECOSYSTEMS\OneDrive\Documents\Chatbot\AIManager\prompt.txt"
#                 )
#             except FileNotFoundError as e:
#                 print(f"File not found: {e}")
#                 continue

#             input_text = (
#                 f"{transcript} Reformat this by removing spelling mistakes and grammatical errors. "
#                 "Ignore repeating words like 'i has the thankyou i has the thankyou'. "
#                 "If the conversation implies the user wants to end it, add 'okay bye' to the output. "
#                 f"Further, here is relevant chat history: {relevant_past_info}."
#                 f"Here is the latest update: {memory_context} Add it in the output if it is relevant."
#                 "Make sure you format the output as it will sound sensible and don't give any solutions just reformat the transcript said by the user."
#             )

#             try:
#                 filtered_response = llm.invoke(input_text)
#             except TypeError as e:
#                 print(f"TypeError: {e}")
#                 print("Ensure that input to LLM is JSON serializable.")
#                 continue

#             response_text = (
#                 f"{query} :- {filtered_response.content} "
#                 f"Here is a summary of the previous conversations: {memory_context}."
#                 f"Don't give output as a conversation; give it as a manager, not a full conversation between you and the user."
#                 f"Make sure you talk as a manager, not with a preamble."
#             )

#             try:
#                 ai_response = llm.invoke(response_text)
#                 print("AI Response:", ai_response.content)
#             except TypeError as e:
#                 print(f"TypeError: {e}")
#                 continue

#             print("Playing Audio")
#             await speak(
#                 text=ai_response.content, 
#                 voice_number=1, rate="+7%", pitch="+10Hz"
#             )

#             store_in_long_term_memory(
#                 user_query=filtered_response.content, 
#                 ai_response=ai_response.content
#             )

#             if check_conversation_end(transcript):
#                 await speak(
#                     text="Thank you, goodbye!", 
#                     voice_number=1, rate="+3%", pitch="+5Hz"
#                 )
#                 break

#     except KeyboardInterrupt:
#         print("Conversation loop stopped.")

# # Main entry point
# if __name__ == "__main__":
#     asyncio.run(Reader())

