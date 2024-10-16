import asyncio
import edge_tts
import io
import pygame

VOICES = [
    'en-IN-NeerjaNeural', 
    'en-IN-PrabhatNeural', 
    'en-CA-ClaraNeural',
    'en-CA-LiamNeural', 
    'en-GB-LibbyNeural', 
    'zh-CN-YunyangNeural',
    'zh-CN-XiaoxiaoNeural'
]

async def amain(text: str, voice: str, rate: str, pitch: str) -> io.BytesIO:
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    audio_data = io.BytesIO()

    # Iterate over the async generator to get the chunks of audio
    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            audio_data.write(chunk['data'])  # Write audio data to BytesIO

    audio_data.seek(0)  # Reset stream position to the start
    return audio_data

async def play_audio(text: str, voice_number: int = 0, rate: str = "+40%", pitch: str = "+10Hz"):
    # Initialize pygame mixer for playing audio
    pygame.mixer.init()

    # Select the voice based on the voice_number provided
    voice = VOICES[voice_number]

    # Fetch audio from edge_tts in memory
    audio_data = await amain(text, voice, rate, pitch)

    # Load the audio stream into pygame for playback
    pygame.mixer.music.load(audio_data, "mp3")

    # Play the audio
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# asyncio.run(play_audio(text="Your String here ", voice_number=0, rate="+10%", pitch="+8Hz"))