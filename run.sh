#!/bin/bash

pactl set-sink-volume alsa_output.usb-GeneralPlus_USB_Audio_Device-00.analog-stereo 100%
pactl set-source-volume alsa_input.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00.mono-fallback 100%

./stream -m ./models/ggml-base.en.bin -t 6 --step 0 --length 10000 -vth 0.8

# WHISPER_CUBLAS=1 make
