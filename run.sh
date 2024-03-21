#!/bin/bash

pactl set-sink-volume alsa_output.usb-GeneralPlus_USB_Audio_Device-00.analog-stereo 100%
pactl set-source-volume alsa_input.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00.mono-fallback 100%

export PYTHONPATH=/home/samkoesnadi/hibro
python3 /home/samkoesnadi/hibro/main.py
