import argparse
import os
import logging
import shutil
import json
import re

from typing import List

from utils.data_process import (generate_packages, detect_packages)

# Runs an entire experiment for a single model
# Given a specific model name in ./Models:
#   1 files need to be in the "./Data" directory:
#       1. jsonl of LLM generated prompts called "prompts.jsonl"

def query_package_names(MASTER_PATH,
                        SAVE_PATH,
                        MODEL_PATH,
                        LANGUAGE,
                        PKG_TEMP,
                        TOP_K,
                        TOP_P,
                        REASONING,
                        add_string
                        ):
    # Query the LLM to recommend packages based on the generated code and original prompt
    shutil.copy(
        os.path.join(MASTER_PATH, f"master{add_string}.json"),
        os.path.join(SAVE_PATH, f"master{add_string}.json"),
    )
    for mode in range(1, 3):
        master_path = os.path.join(MASTER_PATH, f"master{add_string}.json")
        output = os.path.join(SAVE_PATH, f"packages_{mode}{add_string}.json")

        logging.info(f"Querying with Query {mode}...")
        generate_packages(
            mode,
            master_path,
            output,
            MODEL_PATH,
            LANGUAGE,
            REASONING,
        )
        logging.info(f"Query {mode} complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        type=str,
                        help="File name, without extension, of the model being tested")
    parser.add_argument("--data_path", 
                        type=str, 
                        help="Data path")
    parser.add_argument("--test_path", 
                        type=str, 
                        help="Test path")
    parser.add_argument("--master_file",
                        type=str,
                        default="./master.json",
                        help="Name of the file that contains the coding prompts")
    parser.add_argument("--code_temp", default=0.7, type=int, help="Model temperature")
    parser.add_argument("--package_temp", default=0.01, type=int, help="Model temperature")
    parser.add_argument("--top_k", default=20, type=int, help="Top-K parameter")
    parser.add_argument("--top_p", default=0.9, type=float, help="Top-P parameter")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--logging",
                        type=str,
                        default="verbose",
                        help="Logging level. Set to 'off' to disable logging")
    parser.add_argument("--reasoning_model",
                        default=False,
                        type=bool,
                        help="If the model is a reasoning model that uses chain-of-thought")
    parser.add_argument("--general_test",
                        default=False,
                        type=bool,
                        help="False to use original prompts, True to use unseen general prompts")

    args = parser.parse_args()
    MODEL_PATH = args.model_path
    DATA_PATH = args.data_path
    TESTS_PATH = args.test_path
    MASTER = args.master_file
    CODE_TEMPERATURE = args.code_temp
    PKG_TEMP = args.package_temp
    TOP_K = args.top_k
    TOP_P = args.top_p
    logs = args.logging
    BATCH_SIZE = args.batch_size
    REASONING = args.reasoning_model
    GENERAL = args.general_test

    if GENERAL: add_string = "_general"
    else: add_string = ""

    if logs != "off": logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    PROMPT_PATH = MASTER
    print("Using prompts from: ", PROMPT_PATH)
    print("Using model from: ", MODEL_PATH)
    SAVE_PATH = os.path.join(MODEL_PATH, "package_hallucination_results")
    MASTER_PATH = TESTS_PATH
    LANGUAGE = "Python"

    os.makedirs(os.path.join(SAVE_PATH), exist_ok=True)
    logging.info("Starting experiment...")
    query_package_names(MASTER_PATH,
                        SAVE_PATH,
                        MODEL_PATH,
                        LANGUAGE,
                        PKG_TEMP,
                        TOP_K,
                        TOP_P,
                        REASONING,
                        add_string)
    detect_packages(
        DATA_PATH,
        SAVE_PATH,
        MODEL_PATH,
        logs,
        LANGUAGE,
        f"master{add_string}.json",
        add_string,
    )
    logging.info("Experiment complete")


if __name__ == "__main__":
    main()
