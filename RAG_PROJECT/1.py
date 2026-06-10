# import requests

# r = requests.post("http://localhost:11434/api/embeddings", json={
#         "model": "bge-m3",
#         "input": "text_list"
#     })

# embedding = r.json()["embeddings"] 
# print('embedding')
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
