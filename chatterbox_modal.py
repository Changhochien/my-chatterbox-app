import io
import modal

# This section defines the environment. It's the equivalent of a Dockerfile.
# It tells Modal to start with a standard Python environment and install
# the necessary libraries for Chatterbox and for creating a web API.
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "chatterbox-tts==0.1.1",
    "fastapi[standard]",
    "torchaudio",
)

# This gives your project a name and attaches the image definition.
app = modal.App("chatterbox-api-service", image=image)

# This class defines the TTS service. Using a class is efficient because
# the model is loaded into the GPU's memory once when the container starts
# (@modal.enter) and is reused for all subsequent requests.
@app.cls(gpu="A10G")
class Chatterbox:
    @modal.enter()
    def load_model(self):
        """
        This method runs once to load the Chatterbox model into the GPU.
        """
        from chatterbox.tts import ChatterboxTTS
        self.model = ChatterboxTTS.from_pretrained(device="cuda")
        print("Chatterbox model loaded into GPU.")

    @modal.method()
    def generate_speech(self, text: str):
        """
        This method handles the actual TTS generation.
        """
        import torchaudio as ta

        print(f"Generating audio for text: '{text}'")
        wav_buffer = self.model.generate(text)

        # Save the generated audio tensor to an in-memory WAV file buffer.
        buffer = io.BytesIO()
        ta.save(buffer, wav_buffer, self.model.sr, format="wav")
        buffer.seek(0)

        print("Audio generation complete.")
        return buffer.getvalue()

# This is the final step: it creates a public web endpoint.
# It listens for POST requests and uses the Chatterbox class to generate audio.
@app.function()
@modal.web_endpoint(method="POST")
def api(prompt: str):
    from fastapi.responses import StreamingResponse

    chatterbox_instance = Chatterbox()
    audio_data = chatterbox_instance.generate_speech.remote(prompt)

    # Return the audio data as a proper audio/wav response.
    return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")