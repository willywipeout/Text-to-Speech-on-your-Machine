from flask import Flask, request, send_file
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

app = Flask(__name__)

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

@app.route('/synthesize', methods=['POST'])
def synthesize_audio():
    # Get text from request
    text = request.json.get('text', '')
    
    # Load embeddings dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    
    # Generate speech
    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})
    
    # Save audio as WAV file
    sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"], format='WAV')
    
    # Return the audio file
    return send_file("speech.wav", mimetype="audio/wav")

if __name__ == '__main__':
    app.run(debug=True)
