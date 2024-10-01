# This example is adapted from
# https://platform.openai.com/docs/guides/speech-to-text

from openai import OpenAI
client = OpenAI()

#audio_file= open("/path/to/file/harvard.wav", "rb")
audio_file= open("./harvard.wav", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file
)
print(transcription.text)
