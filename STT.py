from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf  # for reading audio files
import torch

# Load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
model.config.forced_decoder_ids = None

# Read file
# audio_file_path = "/workspaces/codespaces-blank/speech.wav" 
# audio_file_path = "/workspaces/codespaces-blank/money-in-savings.wav"
# audio_file_path = "/workspaces/codespaces-blank/credit-score.wav" 
audio_file_path = "/workspaces/codespaces-blank/accessing-account.wav" 
audio_array, sampling_rate = sf.read(audio_file_path)

# Process audio file
input_features = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features 

# Generate token ids
predicted_ids = model.generate(input_features)

# Decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print("Transcription:", transcription)
