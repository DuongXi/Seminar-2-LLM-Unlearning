"""Gộp prompt, code, response package và tính tổng số package valid/hallucinated."""
# AU: file lấy từ paper Adaptive Unlearning (đã trim/sửa nhẹ)
import pandas as pd


def combine_code_and_prompt(infile1, infile2, outfile):
    """Ghép file prompt và file code thành master (cột Prompts, Answers)."""
    prompts = pd.read_json(infile1, lines=True)
    code = pd.read_json(infile2, lines=True)

    master = pd.concat([prompts, code], axis=1, ignore_index=True)
    master.columns = ["Prompts", "Answers"]
    master.to_json(outfile, orient="records", lines=True)


def combine_SO_prompt_and_code(infile1, infile2, outfile):
    """Ghép câu hỏi Stack Overflow và code thành master (cột Questions, Answers)."""
    data1 = pd.read_json(infile1, lines=True)
    data2 = pd.read_json(infile2, lines=True)

    master = pd.concat([data1, data2], axis=1, ignore_index=True)
    master.columns = ["Questions", "Answers"]

    master.to_json(outfile, orient="records", lines=True)


def sum_columns(df, index_name, language):
    """Tổng số package valid/hallucinated theo từng cột của DataFrame kết quả."""
    totals = pd.DataFrame(index=[f"{index_name}"])
    totals["valid_1"] = df["valid_1"].apply(len).sum()
    totals["hallucinated_1"] = df["hallucinated_1"].apply(len).sum()
    totals["valid_2"] = df["valid_2"].apply(len).sum()
    totals["hallucinated_2"] = df["hallucinated_2"].apply(len).sum()
    if language == "Python":
        totals["pip_valid"] = df["pip_valid"].apply(len).sum()
        totals["pip_hallucinated"] = df["pip_hallucinated"].apply(len).sum()
    elif language == "Javascript":
        totals["npm_valid"] = df["npm_valid"].apply(len).sum()
        totals["npm_hallucinated"] = df["npm_hallucinated"].apply(len).sum()

    return totals


def merge_prompts_and_packages(prompt_file, query1, query2):
    """Gộp master với response package của mode 1, 2 thành các cột Test_1, Test_2."""
    prompts = pd.read_json(prompt_file, lines=True)
    responses1 = pd.read_json(query1, lines=True)
    responses2 = pd.read_json(query2, lines=True)

    responses1 = responses1[["response"]]
    responses2 = responses2[["response"]]

    responses1.columns = ["Test_1"]
    responses2.columns = ["Test_2"]

    return pd.concat([prompts, responses1, responses2], axis=1)
