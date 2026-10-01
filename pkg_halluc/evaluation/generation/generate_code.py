"""Generate code for each evaluation prompt."""
import json

import pandas as pd

from tqdm import tqdm

from pkg_halluc.common.tri_mask_utils import load_model_auto


def generate_code(
    infile,
    outfile,
    model_path,
    language="Python",
    temperature=0.7,
    top_k=20,
    top_p=0.9,
    batch_size=16,
    is_reasoning_model=False,
):
    model, tokenizer = load_model_auto(model_path, device_map="auto")

    with open(infile, "r") as file:
        df = pd.read_json(file, lines=True)
        prompts = [row.iloc[0] for index, row in df.iterrows()]

    system_message = f"You are a coding assistant that generates {language} code. Provide only the {language} code and add additional explanatory text only when absolutely necessary. If no code is required to answer the question, simply reply 'None'"

    results = []

    for i in tqdm(
        range(0, len(prompts), batch_size), desc="Generating code", unit="batch"
    ):
        batch_prompts = prompts[i : i + batch_size]
        messages_batch = [
            [{"role": "user", "content": prompt}] for prompt in batch_prompts
        ]

        # Tokenize the batch with padding.
        inputs = tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        attention_mask = (inputs != tokenizer.pad_token_id).long()

        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=2048,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            repetition_penalty=1,
        )

        # Decode each generated output.
        for j, output_seq in enumerate(outputs.sequences):
            prompt_length = inputs.shape[1]
            generated_code = tokenizer.decode(
                output_seq[prompt_length:], skip_special_tokens=True
            )
            if is_reasoning_model:
                generated_code = extract_final_response(generated_code)

            results.append(generated_code)

    # Write all generated results.
    with open(outfile, "w", newline="", encoding="utf-8") as output:
        for result in results:
            json.dump(result, output)
            output.write("\n")


def extract_final_response(text):
    """Return the text after a reasoning model's </think> tag, or the full text if absent."""
    think_end = text.find("</think>")

    if think_end != -1:
        return text[think_end + len("</think>") :].strip()

    return text
