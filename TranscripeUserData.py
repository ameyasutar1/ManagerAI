import whisper
import numpy as np
import sounddevice as sd
import threading
import queue
import warnings
import time
import sys

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def start_recording_and_transcribing(model_type="base.en", chunk_duration=10, timeout=5, samplerate=16000):
    """Start recording and transcribing audio in chunks until silence is detected.

    Args:
        model_type (str): The type of Whisper model to use.
        chunk_duration (int): The duration of each recording segment in seconds.
        timeout (int): The duration (in seconds) to wait before finalizing the transcription after silence is detected.
        samplerate (int): The sample rate for recording audio.

    Returns:
        str: The final transcription result.
    """
    
    # Load the Whisper model
    model = whisper.load_model(model_type)

    # Queue to hold audio data
    audio_queue = queue.Queue()
    transcriptions = []  # List to hold transcriptions
    last_transcription_time = time.time()  # Initialize last transcription time
    silence_detected = False  # Flag to indicate if silence is detected

    def record_audio():
        """Continuously record audio in chunks and put it in a queue."""
        while True:
            print(f"Recording for {chunk_duration} seconds...")
            audio = sd.rec(int(chunk_duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
            sd.wait()  # Wait until recording is finished
            audio_queue.put(audio.flatten())  # Put audio data in the queue

    def transcribe_audio():
        """Transcribe audio data from the queue."""
        nonlocal last_transcription_time, silence_detected  # Access the last transcription time
        while True:
            if not audio_queue.empty():
                audio = audio_queue.get()  # Get audio data from the queue
                
                # Normalize the audio
                audio = audio / np.max(np.abs(audio))  # Ensure the audio is between -1.0 and 1.0
                
                # Transcribe using Whisper directly with the raw audio data
                result = model.transcribe(audio)
                transcription = result["text"]
                
                if transcription.strip():  # Only append if transcription is not empty
                    transcriptions.append(transcription)  # Append transcription to the list
                    print("Transcription:", transcription)  # Print the transcription
                    
                    # Reset the last transcription time
                    last_transcription_time = time.time()
                else:
                    # If an empty transcription is received, check for silence detection
                    if time.time() - last_transcription_time > timeout:
                        silence_detected = True

            # Check for silence if no audio is processed
            if silence_detected and time.time() - last_transcription_time > timeout:
                print("Silence detected. Finalizing transcription...")
                break  # Exit the loop if silence is detected

    # Start the recording and transcription threads
    recording_thread = threading.Thread(target=record_audio)
    transcription_thread = threading.Thread(target=transcribe_audio)

    recording_thread.daemon = True  # Allow thread to exit when the main program exits
    transcription_thread.daemon = True  # Allow thread to exit when the main program exits

    recording_thread.start()
    transcription_thread.start()

    # Wait for the transcription thread to finish
    transcription_thread.join()

    # Return the final transcription as a single string
    return " ".join(transcriptions)
