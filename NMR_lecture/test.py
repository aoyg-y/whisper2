import torch
import torch_directml
import whisper
import json

dml = torch_directml.device()
model = whisper.load_model("small")#.to(dml) #モデル指定

result = model.transcribe("./20250905.m4a", verbose=True, fp16=False, language="ja") #ファイル指定
print(result['text'])

f = open('transcription2025090520250905202509052025090520250905202509052025090520250905202509052025090520250905202509052025090520250905202509052025090520250905202509052025090520250905.txt', 'w', encoding='UTF-8')
f.write(json.dumps(result['text'], sort_keys=True, indent=4, ensure_ascii=False))
f.close()
