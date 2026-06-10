import whisper
import json

model = whisper.load_model("turbo")
result = model.transcribe(audio = "audios/Ch0.mp3") 

chunks=[]
for segment in result["segments"]:
    chunks.append({"start": segment["start"], "end": segment["end"], "text": segment["text"],})

with open("output.json", "w") as f:
    json.dump(chunks, f)