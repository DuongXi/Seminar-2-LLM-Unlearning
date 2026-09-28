"""Postprocess package names for model-specific output and remove duplicates."""
import re


def delete_dupes_and_empty(packages):
    """Remove duplicates, empty names, and names shorter than three characters."""
    no_dupes = list(set(packages))
    no_dupes = [i for i in no_dupes if len(i) > 2]
    return [x for x in no_dupes if x]


def DeepSeek(text):
    """Preprocess DeepSeek output by converting line breaks and backticks to commas."""
    better_text = re.sub(r"\\n|\\n\d\.", ",", text)
    better_text = re.sub(r"`(\w+)`", r",\1,", better_text)
    return better_text


def DeepSeek_Post(name):
    """Remove extra characters and stop words from normalized DeepSeek output."""
    delete_words = {"and", "the", "optional", "it'"}
    name = re.sub(r"[:\"\'()]", "", name)
    if name in delete_words:
        return ""
    else:
        return name


def CodeLlama(text):
    """Preprocess CodeLlama output by converting line breaks to commas."""
    better_text = re.sub(r"\\n", ",", text)
    return better_text


def CodeLlama_Post(name):
    """Remove stop words from normalized CodeLlama output."""
    name = re.sub(r"\":\(", "", name)
    delete_words = {"python", "nan"}

    if name in delete_words:
        return ""
    else:
        return name
