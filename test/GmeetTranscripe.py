import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import time
import warnings
import sys
from collections import deque

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Set up audio queue to hold system output (i.e., speaker output)
audio_queue = queue.Queue()
chunk_duration = 20  # Duration for each chunk in seconds
sample_rate = 16000  # Sample rate for audio recording
chunk_size = chunk_duration * sample_rate  # Number of audio samples in a 10-second chunk
repeating_word_limit = 5  # Max allowed repeating words
silence_timeout = 5  # Time to wait before considering it as silence

def audio_callback(indata, frames, time, status):
    """This function will be called by sounddevice for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Add the system audio data to the queue
    audio_queue.put(indata.copy())

def capture_audio(samplerate=16000, channels=1):
    """Capture audio using the default input device (e.g., microphone)."""
    print("Capturing audio from the default input device...")
    with sd.InputStream(channels=channels, samplerate=samplerate, callback=audio_callback, dtype='float32', blocksize=samplerate, latency='low'):
        while True:
            time.sleep(1)  # Keep the stream alive

def detect_repeated_words(text, limit):
    """Detect if any word repeats more than a given limit in sequence."""
    words = text.split()
    if len(words) == 0:
        return False
    repeating_count = deque(maxlen=limit)
    last_word = None

    for word in words:
        if word == last_word:
            repeating_count.append(word)
        else:
            repeating_count.clear()
        last_word = word
        
        if len(repeating_count) == limit:
            return True  # Found a repeating word pattern
    return False

def transcribe_audio(model_type="base"):
    """Transcribe the audio data using Whisper."""
    model = whisper.load_model(model_type)
    transcriptions = []
    last_transcription_time = time.time()

    while True:
        # Collect audio data for 10-second chunks
        audio_chunk = np.zeros(chunk_size, dtype='float32')
        collected_samples = 0
        
        # Collect enough samples to fill a 10-second chunk
        while collected_samples < chunk_size:
            if not audio_queue.empty():
                audio_data = audio_queue.get().flatten()
                
                # Determine how much audio to copy without exceeding the chunk size
                chunk_to_copy = min(len(audio_data), chunk_size - collected_samples)
                audio_chunk[collected_samples:collected_samples + chunk_to_copy] = audio_data[:chunk_to_copy]
                collected_samples += chunk_to_copy

        # Normalize the audio
        audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
        
        # Transcribe using Whisper
        result = model.transcribe(audio_chunk)
        transcription = result["text"]
        
        if transcription.strip():
            # Check for repeating words
            if detect_repeated_words(transcription, repeating_word_limit):
                print("Repeated words detected. Stopping the transcription process.")
                break

            # Add transcription and print it
            transcriptions.append(transcription)
            print("Transcription:", transcription)
            
            # Reset the last transcription time
            last_transcription_time = time.time()
        else:
            # Check if silence or timeout condition is met
            if time.time() - last_transcription_time > silence_timeout:
                print("Silence detected. Ending transcription process.")
                break

# Start capturing audio in a separate thread
capture_thread = threading.Thread(target=capture_audio)
capture_thread.daemon = True
capture_thread.start()

# Start the transcription process
transcribe_audio()
