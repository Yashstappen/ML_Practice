import os
import subprocess
files = os.listdir("videos")

for file in files:
    num = file.split("- ")[1].split(" ")[0]
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{num}.mp3"])
