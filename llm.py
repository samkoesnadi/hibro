import json
import string

from llama_cpp import Llama

from utils import say, control_lamp_function

TOKEN_LENGTH = 256

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
            "description": "The user can use this function by setting the on_off argument.",
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
    functionList += json.dumps(value["metadata"], separators=(',', ': '))

printable = set(string.printable)

def run_llm(prompt):
    output = llm(
f"""<s><<SYS>>I am a home assistant that switch on and off lamp. When using function, it outputs json with "function" and "argument"<</SYS>> <FUNCTION>{functionList}</FUNCTION>
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
        output = output.lower()
        output = json.loads(output)
    except json.decoder.JSONDecodeError:
        output = ''.join(filter(lambda x: x in printable, output))
        say(output)
        return

    if "function" in output:
        arguments = output["arguments"]
        if isinstance(output["arguments"], list):
            arguments = {argument["name"]: argument["value"] for argument in output["arguments"]}
        functionsDict[output["function"]]["function"](
            **arguments)
        return
    elif "error" in output:
        say("Error: " + output["error"])
    else:
        say("There is something wrong with reading the llm json output.")
