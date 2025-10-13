import os
import re
import pandas as pd
import json


def extract_map50_from_log(filepath, keyword):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(rf'{keyword}\s*-->\s*map50\s*:\s*([\d.]+)', content)
    return float(match.group(1)) * 100 if match else -1


def extract_map50_from_json(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        data = json.load(f)
    return float(data["best_all_map50"]) * 100


def collect_results_all_tasks(base_dir):
    all_results = []

    for task in os.listdir(base_dir):  # 遍历如 DINO_HRRSD_to_SSDD
        task_path = os.path.join(base_dir, task)
        if not os.path.isdir(task_path):
            continue

        for setting in os.listdir(task_path):  # 遍历具体实验配置
            setting_path = os.path.join(task_path, setting)
            if not os.path.isdir(setting_path):
                continue

            # 日志路径
            pre_train_log = os.path.join(setting_path, 'pre_train', 'log_best.txt')
            pre_teacher_map50 = extract_map50_from_log(pre_train_log, 'best_teacher')
            self_model_map50 = extract_map50_from_json(os.path.join(setting_path, 'self_train', 'result_summary.json'))
            # self_model_map50_3 = extract_map50_from_json(os.path.join(setting_path, 'self_train_thr_0.3', 'result_summary.json'))
            # self_model_map50_4 = extract_map50_from_json(os.path.join(setting_path, 'self_train_thr_0.4', 'result_summary.json'))
            # self_model_map50_5 = extract_map50_from_json(os.path.join(setting_path, 'self_train_thr_0.5', 'result_summary.json'))
            # self_model_map50_5 = extract_map50_from_json(os.path.join(setting_path, 'self_train_thr_0.6', 'result_summary.json'))

            all_results.append({
                'task': task,
                'setting': setting,
                'pre_train_best_teacher_map50': pre_teacher_map50,
                'self_model_map50': self_model_map50,
                # 'self_train_thr_0.3': self_model_map50_3,
                # 'self_train_thr_0.4': self_model_map50_4,
                # 'self_train_thr_0.5': self_model_map50_5,
                # 'self_train_thr_0.6': self_model_map50_5,
            })


    return pd.DataFrame(all_results)

def collect_results_all_tasks_teacher(base_dir):
    all_results = []

    for task in os.listdir(base_dir):  # 遍历如 DINO_HRRSD_to_SSDD
        task_path = os.path.join(base_dir, task)
        if not os.path.isdir(task_path):
            continue

        for setting in os.listdir(task_path):  # 遍历具体实验配置
            setting_path = os.path.join(task_path, setting)
            if not os.path.isdir(setting_path):
                continue

            # 日志路径
            self_model_map50 = extract_map50_from_json(os.path.join(setting_path, 'result_summary.json'))
            if self_model_map50 is not None:
                all_results.append({
                    'task': task,
                    'setting': setting,
                    'self_model_map50': self_model_map50,
                })

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    root_dir = 'reproduction/'
    df = collect_results_all_tasks(root_dir)
    df.to_excel('experiment_summary.xlsx', index=False)  # 保存为Excel
    print("实验结果已保存为 experiment_summary.xlsx")
