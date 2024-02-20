import numpy as np
from utils import say
from llm import run_llm
import whisper
from scipy.io.wavfile import write as wavwrite

import os
import time

from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector

from eff_word_net.audio_processing import Resnet50_Arc_loss

from eff_word_net import samples_loc


import torch
vadModel, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=True)

def checkVad(frame, window_size_samples = 1536, vadThreshold = 0.7):
    for i in range(0, len(frame), window_size_samples):
        chunk = frame[i: i+ window_size_samples]
        if len(chunk) < window_size_samples:
            break
        new_confidence = vadModel(torch.from_numpy(chunk), 16000).item()
        
        if new_confidence >= vadThreshold:
            return True
    return False

hotword_base_model = Resnet50_Arc_loss()
mycroft_hw = HotwordDetector(
    hotword="mycroft",
    model = hotword_base_model,
    reference_file=os.path.join(samples_loc, "mycroft_ref.json"),
    threshold=0.7,
    relaxation_time=2
)

sliding_window_secs = 0.75
mic_stream = SimpleMicStream(
    window_length_secs=1.5,
    sliding_window_secs=sliding_window_secs,
)

whisperModel = whisper.load_model("base.en")

def restart_mic_stream():
    mic_stream.close_stream()
    mic_stream.start_stream()
    mic_stream.getFrame()


if __name__ == "__main__":
    mic_stream.start_stream()

    print("Say Mycroft ")

    recordState = False
    recordedArray = []
    lastHotwordSecond = time.time()
    while True :
        frame = mic_stream.getFrame().astype(np.int16)

        if recordState is False:
            result = mycroft_hw.scoreFrame(frame.copy())
            if result==None :
                #no voice activity
                continue
            if(result["match"]):
                say("listening.")
                restart_mic_stream()
                recordState = True
                lastHotwordSecond = time.time()
        else:
            # The actual length is every sliding_window_secs time
            frame = frame[int(sliding_window_secs * 16000):].copy()
            normalizedframe = (
                frame.astype('float32') * (1 / 32768.0))
            vadExists = checkVad(normalizedframe)

            if vadExists:
                lastHotwordSecond = time.time()

            if (time.time() - lastHotwordSecond) < 2.0:
                recordedArray.append(normalizedframe)
            else:
                say("processing")
                # wavwrite('test.wav', 16000, np.concatenate(recordedArray, dtype=np.float32))
                result = whisperModel.transcribe(
                    np.concatenate(recordedArray, dtype=np.float32),
                    language="english",
                    fp16=torch.cuda.is_available(),
                )
                print(result["text"].strip())
                run_llm(result["text"].strip())
                recordState = False
                recordedArray = []
                restart_mic_stream()

        # if vadExists or (recordState == True and (time.time() - lastHotwordSecond) < 0.5):
        #     print("vad")
        #     if recordState:
        #         recordedArray.append(normalizedframe)
        #     else:
        #         result = mycroft_hw.scoreFrame(frame)
        #         if result==None :
        #             #no voice activity
        #             continue
        #         if(result["match"]):
        #             say("listening")
        #             recordState = True
        #             lastHotwordSecond = time.time()
        # else:
        #     if recordState:
        #         say("transcribing")
        #         print(np.concatenate(recordedArray).shape)
        #         print(np.concatenate(recordedArray).dtype)
        #         result = whisperModel.transcribe(np.concatenate(recordedArray))
        #         say("running llm")
        #         print(result["text"])
        #         run_llm(result["text"])
        #         recordState = False
        #         recordedArray = []
