import whisper
import json
import os

model = whisper.load_model("turbo")

audios = os.listdir("audios")

for audio in audios:
    name = audio[:-4]
    result = model.transcribe(audio = f"audios/{audio}") 

chunks=[]
for segment in result["segments"]:
    chunks.append({"Name":name ,"Start": segment["start"], "End": segment["end"], "Text": segment["text"],})

chunks_with_metadata = {"chunks" : chunks, "text" : result["text"]}

with open(f"jsons/{audio}.json", "w") as f:
    json.dump(chunks_with_metadata, f)