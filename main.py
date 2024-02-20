import numpy as np
from utils import say
from llm import run_llm
import whisper

import os
import time

from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector

from eff_word_net.audio_processing import Resnet50_Arc_loss

from eff_word_net import samples_loc


import torch
model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=True)

def checkVad(frame, window_size_samples = 1536, vadThreshold = 0.7):
    for i in range(0, len(frame), window_size_samples):
        chunk = frame[i: i+ window_size_samples].astype('float32') / 32768
        print(chunk)
        if len(chunk) < window_size_samples:
            break
        new_confidence = model(torch.from_numpy(chunk), 16000).item()
        
        if new_confidence >= vadThreshold:
            return True
    return False

base_model = Resnet50_Arc_loss()

mycroft_hw = HotwordDetector(
    hotword="mycroft",
    model = base_model,
    reference_file=os.path.join(samples_loc, "mycroft_ref.json"),
    threshold=0.7,
    relaxation_time=2
)

mic_stream = SimpleMicStream(
    window_length_secs=1.5,
    sliding_window_secs=0.75,
)

model = whisper.load_model("tiny.en")


if __name__ == "__main__":
    mic_stream.start_stream()

    print("Say Mycroft ")

    recordState = False
    recordedArray = []
    while True :
        frame = mic_stream.getFrame()
        normalizedframe = frame
        vadExists = checkVad(normalizedframe)

        if vadExists:
            if recordState:
                recordedArray.append(normalizedframe)
            else:
                result = mycroft_hw.scoreFrame(frame)
                if result==None :
                    #no voice activity
                    continue
                if(result["match"]):
                    say("listening")
                    recordState = True
        else:
            if recordState:
                say("transcribing")
                result = model.transcribe(np.concatenate(recordedArray))
                say("running llm")
                print(result["text"])
                run_llm(result["text"])
                recordState = False
                recordedArray = []
