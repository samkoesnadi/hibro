import json
import string

from llama_cpp import Llama

from utils import say, control_lamp_function

TOKEN_LENGTH = 1024

llm = Llama(
    model_path="/home/samkoesnadi/models/Llama-2-7b-chat-hf-function-calling-v3.Q5_K_M.gguf",
    chat_format="llama-2",
    n_ctx=TOKEN_LENGTH,  # The max sequence length to use - note that longer sequence lengths require much more resources
    # n_threads=6,            # The number of CPU threads to use, tailor to your system and the resulting performance
    n_gpu_layers=-1         # The number of layers to offload to GPU, if you have GPU acceleration available
)


functionsDict = {
    "control_lamp": {
        "function": control_lamp_function,
        "metadata": {
            "function": "control_lamp",
            "description": "Turn on or off the lamp. This allows users to turn on or off the lamp which is determine by the on_off argument.",
            "arguments": [
                {
                    "name": "on_off",
                    "type": "bool",
                    "description": "boolean 1 if on, and boolean 0 if off"
                }
            ]
        }
    }
}

functionList = ''
for key, value in functionsDict.items():
    functionList += json.dumps(value["metadata"], indent=1, separators=(',', ': '))

printable = set(string.printable)

def run_llm(prompt):
    output = llm(
f"""<s> <FUNCTIONS>{functionList}</FUNCTIONS>
[INST] {prompt} [/INST]
""", # Prompt
        max_tokens=TOKEN_LENGTH,  # Generate up to 512 tokens
        stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
        echo=False,        # Whether to echo the prompt
        temperature=0.0,
        repeat_penalty=1.1
    )

    print(output)

    output = output["choices"][0]["text"]
    output = output.strip()

    try:
        output = json.loads(output)
    except json.decoder.JSONDecodeError:
        output = ''.join(filter(lambda x: x in printable, output))
        say(output)
        return

    if "function" in output:
        functionsDict[output["function"]]["function"](
            **output["arguments"])
        return
    else:
        say("There is something wrong with the program.")
