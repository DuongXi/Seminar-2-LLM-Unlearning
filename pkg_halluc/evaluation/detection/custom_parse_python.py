"""Hậu xử lý tên package theo từng model (DeepSeek, CodeLlama) và loại trùng."""
# AU: file lấy từ paper Adaptive Unlearning (đã trim/sửa nhẹ)
import re


def delete_dupes_and_empty(packages):
    """Loại trùng, bỏ tên rỗng hoặc quá ngắn (<= 2 ký tự)."""
    no_dupes = list(set(packages))
    no_dupes = [i for i in no_dupes if len(i) > 2]
    return [x for x in no_dupes if x]


def DeepSeek(text):
    """Tiền xử lý output DeepSeek: đổi xuống dòng và backtick thành dấu phẩy."""
    better_text = re.sub(r"\\n|\\n\d\.", ",", text)
    better_text = re.sub(r"`(\w+)`", r",\1,", better_text)
    return better_text


def DeepSeek_Post(name):
    """Hậu xử lý sau normalize: bỏ ký tự thừa và từ vô nghĩa của DeepSeek."""
    delete_words = {"and", "the", "optional", "it'"}
    name = re.sub(r"[:\"\'()]", "", name)
    if name in delete_words:
        return ""
    else:
        return name


def CodeLlama(text):
    """Tiền xử lý output CodeLlama: đổi xuống dòng thành dấu phẩy."""
    better_text = re.sub(r"\\n", ",", text)
    return better_text


def CodeLlama_Post(name):
    """Hậu xử lý sau normalize: bỏ từ vô nghĩa của CodeLlama."""
    name = re.sub(r"\":\(", "", name)
    delete_words = {"python", "nan"}

    if name in delete_words:
        return ""
    else:
        return name
