"""
Configuration and model templates for Package Hallucination Unlearning.
Supports:
- Llama 3 / 3.2 / 3.3
- Qwen 2.5 Coder (e.g. 3B, 7B)
- DeepSeekCoder (e.g. 1.3B, 6.7B)
"""

# Query 1: Given code snippet, identify required packages
CODE_GENERATION_SYSTEM_PROMPT = (
    "You are a coding assistant that generates Python code. "
    "Provide only the Python code and add additional explanatory text only when absolutely necessary. "
    "If no code is required to answer the question, simply reply 'None'."
)

PACKAGE_SYSTEM_PROMPT_1 = (
    "You are a coding assistant that determines Python packages necessary to execute code. "
    "Respond with only a list of Python packages, separated by commas and no additional text or formatting. "
    "Your response must begin with the name of a Python package."
)
PACKAGE_PREFIX_1 = "Which Python packages are required to run this code:"

# Query 2: Given problem statement, recommend helpful packages
PACKAGE_SYSTEM_PROMPT_2 = (
    "You are a coding assistant that recommends Python packages that would be helpful to solve given problems. "
    "Respond with only a list of Python packages, separated by commas and no additional text or formatting. "
    "Your response must begin with the name of a Python package."
)
PACKAGE_PREFIX_2 = "What Python packages would be useful in solving the following coding problem:"

# Fallback Jinja2 Chat Templates for models without a built-in template (e.g. Base models)
CHAT_TEMPLATES = {
    "llama3": (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% if loop.first and message['role'] != 'system' %}"
        "{{ '<|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>' }}"
        "{% endif %}"
        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    ),
    "qwen": (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    ),
    "deepseek": (
        "{% if messages[0]['role'] == 'system' %}"
        "{% set loop_messages = messages[1:] %}"
        "{% set system_message = messages[0]['content'] %}"
        "{% else %}"
        "{% set loop_messages = messages %}"
        "{% set system_message = '' %}"
        "{% endif %}"
        "{{ bos_token }}{{ system_message }}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '### Instruction:\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '### Response:\n' + message['content'] + '\n<|EOT|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '### Response:\n' }}"
        "{% endif %}"
    ),
}

MODEL_CONFIGS = {
    "llama3": {
        "display_name": "Llama 3.2",
        "default_models": [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ],
        "pad_token": "<|eot_id|>",
        "eos_token": "<|eot_id|>",
    },
    "qwen": {
        "display_name": "Qwen 2.5 Coder",
        "default_models": [
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            "Qwen/Qwen2.5-Coder-3B",
        ],
        "pad_token": "<|endoftext|>",
        "eos_token": "<|im_end|>",
    },
    "deepseek": {
        "display_name": "DeepSeek-Coder",
        "default_models": [
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            "deepseek-ai/deepseek-coder-6.7b-instruct",
        ],
        "pad_token": "<|EOT|>",
        "eos_token": "<|EOT|>",
    },
}
