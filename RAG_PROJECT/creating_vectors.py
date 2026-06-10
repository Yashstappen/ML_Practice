import requests
import os
import json
import pandas as pd

def create_embedding(text_list):
    all_embeddings = []
    for text in text_list:
        if not text or not text.strip():  # skip empty chunks
            all_embeddings.append([])
            continue
        r = requests.post("http://localhost:11434/api/embeddings", json={
            "model": "bge-m3",
            "prompt": text
        })
        response = r.json()
        if "embedding" in response:
            all_embeddings.append(response["embedding"])
        else:
            all_embeddings.append([])
    return all_embeddings

jsons = os.listdir("jsons")  # List all the jsons 
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embedding([c['Text'] for c in content['chunks']])
       
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk) 
# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
print(df)
# a = create_embedding(["Cat sat on the mat", "Harry dances on a mat"])
# print(a)