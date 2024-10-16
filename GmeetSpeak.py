import asyncio
import edge_tts
import io
import numpy as np
import sounddevice as sd
import soundfile as sf

# List of voices available
VOICES = [
    'en-IN-NeerjaNeural', 
    'en-IN-PrabhatNeural', 
    'en-CA-ClaraNeural',
    'en-CA-LiamNeural', 
    'en-GB-LibbyNeural', 
    'zh-CN-YunyangNeural',
    'zh-CN-XiaoxiaoNeural'
]

# Function to fetch audio data using edge_tts
async def amain(text: str, voice: str, rate: str, pitch: str) -> bytes:
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    audio_data = io.BytesIO()

    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            audio_data.write(chunk['data'])  # Write audio data to BytesIO

    return audio_data.getvalue()  # Return audio data directly

# Function to play audio through the virtual audio cable
async def play_audio(text: str, voice_number: int = 0, rate: str = "+40%", pitch: str = "+10Hz"):
    # Select the voice based on the voice_number provided
    voice = VOICES[voice_number]

    # Fetch audio from edge_tts in memory
    audio_bytes = await amain(text, voice, rate, pitch)

    # Read the audio bytes into numpy array directly (avoiding file write)
    audio_data_np, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32')

    # Set the correct device index for VB-Cable (use the appropriate index from your list)
    device_index = 11  # Use DirectSound with index 11, or change to 13 for WASAPI

    # Play audio using sounddevice on the virtual cable
    sd.play(audio_data_np, samplerate=sample_rate, device=device_index)

    # Wait until the audio has finished playing
    sd.wait()

# The main function to be called from another script
async def speak(text: str, voice_number: int = 0, rate: str = "+10%", pitch: str = "+8Hz"):
    await play_audio(text=text, voice_number=voice_number, rate=rate, pitch=pitch)

# # Example of how to call it from another script:
# asyncio.run(speak(text="How are you Vaishnavi ? I am Ching ,A Manager At Eco-soft Global Private Limited , I hope you are doing Well i wanted to Introduce you to my Boss Ameya!", voice_number=3, rate="+10%", pitch="+8Hz"))
# asyncio.run(speak(text="Hi Vaishnavi , My Boss Ameya is the best Man in the world , I am SuckLee his assistant ", voice_number=0, rate="+7%",pitch="+10Hz"))
