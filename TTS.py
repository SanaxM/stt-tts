from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = synthesiser("We are going to win this Dragon's Den AI Hackathon!", forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.mp3", speech["audio"], samplerate=speech["sampling_rate"])