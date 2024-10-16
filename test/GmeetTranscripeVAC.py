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

# Function to compute RMS for loudness monitoring
def compute_rms(audio_chunk):
    """Compute the Root Mean Square (RMS) for the loudness of the audio."""
    return np.sqrt(np.mean(np.square(audio_chunk)))

# Function to detect repetitive phrases or excessive repetitions
def detect_repetitive_phrases(text, limit=5):
    """
    Detect if any phrase or word repeats too many times in sequence.

    Parameters:
    - text: Transcription text.
    - limit: Max allowed repetition of words or phrases.
    
    Returns:
    - True if repetition is detected, False otherwise.
    """
    words = text.split()
    if len(words) == 0:
        return False
    
    # Count repeated phrases
    last_phrase = []
    repeat_count = 0
    max_repeat = limit
    
    for i in range(len(words)):
        current_phrase = words[i:i+3]  # Check for repeating 3-word patterns

        if current_phrase == last_phrase:
            repeat_count += 1
        else:
            repeat_count = 0
        
        last_phrase = current_phrase

        if repeat_count >= max_repeat:
            return True  # Detected too many repetitions

    return False

# Main function to control the audio capture and transcription process
def start_transcription(
    chunk_duration=10, sample_rate=16000, loudness_start_threshold=0.1, 
    loudness_stop_threshold=0.07, repeating_word_limit=5, silence_timeout=1, 
    model_type="base.en", vac_input_device=0, channels=1):
    """
    Function to start the audio capture and transcription process.

    Parameters:
    - chunk_duration: Duration of each chunk to process (in seconds).
    - sample_rate: Sample rate for audio recording (default 16000 Hz).
    - loudness_start_threshold: Loudness threshold to start recording.
    - loudness_stop_threshold: Loudness threshold to stop recording.
    - repeating_word_limit: Maximum allowed repeated words/phrases in transcription.
    - silence_timeout: Timeout for silence detection (in seconds).
    - model_type: Whisper model type (e.g., "base.en").
    - vac_input_device: Input device for capturing audio (e.g., VAC input device).
    - channels: Number of audio channels to record (default 1).

    Returns:
    - Complete transcription text.
    """
    
    audio_queue = queue.Queue()  # Queue to store audio data
    chunk_size = chunk_duration * sample_rate  # Number of audio samples in each chunk
    recording = False  # Flag to indicate if recording is active
    transcriptions = []  # Store transcriptions

    def audio_callback(indata, frames, time, status):
        """Callback to handle incoming audio blocks from sounddevice."""
        if status:
            print(status, file=sys.stderr)
        # Add audio data to the queue
        audio_queue.put(indata.copy())

    def capture_audio():
        """Capture audio from the VAC input device using sounddevice."""
        print("Capturing audio from VAC...")
        with sd.InputStream(channels=channels, samplerate=sample_rate, callback=audio_callback,
                            dtype='float32', blocksize=sample_rate, latency='low', device=vac_input_device):
            while True:
                time.sleep(1)  # Keep the audio stream alive

    def transcribe_audio():
        """Transcribe captured audio using Whisper and monitor loudness."""
        nonlocal recording  # Access the recording flag

        model = whisper.load_model(model_type)
        last_transcription_time = time.time()

        while True:
            # Collect audio data for each chunk
            audio_chunk = np.zeros(chunk_size, dtype='float32')
            collected_samples = 0

            # Collect enough samples to fill one chunk
            while collected_samples < chunk_size:
                if not audio_queue.empty():
                    audio_data = audio_queue.get().flatten()

                    # Determine how much audio to copy without exceeding the chunk size
                    chunk_to_copy = min(len(audio_data), chunk_size - collected_samples)
                    audio_chunk[collected_samples:collected_samples + chunk_to_copy] = audio_data[:chunk_to_copy]
                    collected_samples += chunk_to_copy

            # Normalize the audio chunk
            audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))

            # Compute loudness using RMS
            rms_value = compute_rms(audio_chunk)
            print(f"Loudness (RMS): {rms_value:.4f}")

            # Start recording if loudness exceeds the start threshold
            if rms_value >= loudness_start_threshold:
                if not recording:
                    print("Loudness above start threshold. Starting recording.")
                    recording = True

            # Stop recording if loudness goes below the stop threshold
            if rms_value <= loudness_stop_threshold:
                if recording:
                    print("Loudness below stop threshold. Stopping recording.")
                    recording = False

            # Only transcribe if recording is active
            if recording:
                # Transcribe the audio chunk using Whisper
                result = model.transcribe(audio_chunk)
                transcription = result["text"]

                if transcription.strip():
                    # Detect repeated phrases in transcription
                    if detect_repetitive_phrases(transcription, repeating_word_limit):
                        print("Repetitive phrases detected. Stopping the transcription process.")
                        break

                    # Append transcription to the list and print it
                    transcriptions.append(transcription)
                    print("Transcription:", transcription)

                    # Reset transcription time after each valid transcription
                    last_transcription_time = time.time()

            else:
                print("Recording is off, skipping transcription.")

            # Check if silence timeout is reached
            if time.time() - last_transcription_time > silence_timeout:
                print("Silence detected. Ending transcription process.")
                break

        return " ".join(transcriptions)

    # Start audio capture in a separate thread
    capture_thread = threading.Thread(target=capture_audio)
    capture_thread.daemon = True
    capture_thread.start()

    # Start the transcription process and return the final result
    return transcribe_audio()

# # Example usage: customize parameters as needed
# transcription_result = start_transcription(
#     chunk_duration=10,               # Set duration of audio chunks in seconds
#     sample_rate=16000,               # Set the sample rate
#     loudness_start_threshold=0.1,    # Set loudness threshold to start recording
#     loudness_stop_threshold=0.08,    # Set loudness threshold to stop recording
#     repeating_word_limit=5,          # Limit for detecting repeated phrases
#     silence_timeout=5,               # Silence timeout to stop transcription
#     model_type="base.en",            # Whisper model type (default is "base.en")
#     vac_input_device=0,              # VAC input device (change as per your setup)
#     channels=1                       # Number of audio channels (mono audio)
# )

# # Output the final transcription
# print("Final Transcription:", transcription_result)
