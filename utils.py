import subprocess
import requests
import re

def say(arg):
    for splitted_arg in re.split("\\.|\n", arg):
        splitted_arg = splitted_arg.strip()
        if splitted_arg != "":
            subprocess.run(["say", splitted_arg], check=False, text=False)

def control_lamp_function(on_off):
    api_url = "http://samilamp.local/"
    if on_off:
        arg = "on"
        success_words = "Turning on successful"
    else:
        arg = "off"
        success_words = "Turning off successful"

    response = requests.get(api_url + arg)
    
    if int(response.status_code) == 200:
        say(success_words)
    else:
        say("I can not reach the auto switcher url")
