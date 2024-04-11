# 1
# Script using OpenAI's Whisper for STT and Microsoft's SpeechT5 for TTS

from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import soundfile as sf
import torch

# TTS synthesizer
tts_synthesizer = pipeline("text-to-speech", "microsoft/speecht5_tts")

# STT model and processor
stt_processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="en")
stt_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
stt_model.config.forced_decoder_ids = None

# speaker embeddings dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Read audio file 
audio_file_path = "test.wav"
audio_array, sampling_rate = sf.read(audio_file_path)
input_features = stt_processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features

# Generate token ids
question_ids = stt_model.generate(input_features)
question_text = stt_processor.batch_decode(question_ids, skip_special_tokens=True)[0]

# Azure OpenAI bot


# A fake response
fake_response_text = "Your credit score is three hundred"

# Convert the response
response_audio = tts_synthesizer(fake_response_text, forward_params={"speaker_embeddings": speaker_embedding})

# Save to an audio file
sf.write("fake_response.wav", response_audio["audio"], samplerate=response_audio["sampling_rate"])


#2
# Script using Facebook's Wav2Vec2 for STT and Microsoft's SpeechT5 for TTS

import librosa
import torch
import logging
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, pipeline
from datasets import load_dataset
import soundfile as sf


logging.getLogger("transformers").setLevel(logging.ERROR)

# Load STT
tokenizer_stt = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model_stt = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load TTS
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

# speaker embeddings dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)


speech_input = "What is my credit score?"
speech = synthesiser(speech_input, forward_params={"speaker_embeddings": speaker_embedding})

# Save speech audio
sf.write("test.wav", speech["audio"], samplerate=speech["sampling_rate"])

# Load speech audio
speech_stt, rate_stt = librosa.load("test.wav", sr=16000)

# Tokenize speech input
input_values_stt = tokenizer_stt(speech_stt, return_tensors="pt").input_values

# Store logits (non-normalized predictions)
logits_stt = model_stt(input_values_stt).logits

# Store predicted ids
predicted_ids_stt = torch.argmax(logits_stt, dim=-1)

# Decode the audio to generate text
transcription_stt = tokenizer_stt.decode(predicted_ids_stt[0])

print("Transcription from STT:", transcription_stt)

# Azure OpenAI bot

# fake response 
bot_response = "Your credit score is three hundred"

# Generate speech from response
speech_bot = synthesiser(bot_response, forward_params={"speaker_embeddings": speaker_embedding})

# Save to an audio file
sf.write("bot_response.wav", speech_bot["audio"], samplerate=speech_bot["sampling_rate"])
