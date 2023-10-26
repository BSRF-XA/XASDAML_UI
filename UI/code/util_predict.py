import os
import sys
import pandas as pd
import numpy as np
from tensorflow import keras

def check_path_exist(list_path, W_igore = False):
    """
    检查文件是否存在
    Input  : list of paths
    W_igore : whether to ignore the warning message
              True: ignore the warning message and continue running the code
              False: giving the error message and stop running the code
    Output : None or Error message following the exit of code(W_igore = false) or continue the program(W_igore = True)
    """
    for path in list_path:
        exist = os.path.exists(path)
        if not exist:
            print()
            print(f"** Error!! cannot find file {path}! **")
            print()
            if W_igore:
                pass
            else:
                sys.exit()

def import_data(dir_data, file_feature):
    """
    检查数据是否存在，然后导入数据
    """
    path_dir_data = os.path.join(os.curdir, dir_data)
    path_file_data = os.path.join(path_dir_data, file_feature)
    # check whether the files exist
    # 检查预测文件、原值文件是否存在
    list_path = [path_file_data]
    check_path_exist(list_path)
    # 读取test文件，制成数据集
    data_df = pd.read_csv(path_file_data, header=None, delim_whitespace=True)
    feature_test = data_df.values
    # print(f"feature_test.shape = {feature_test.shape}")
    # print(f"feature_test = {feature_test}")
    return feature_test

def import_model(dir_model, file_model):
    """
    检查模型是否存在，然后导入模型
    """
    path_dir_model = os.curdir + '/' + dir_model
    # print("path_dir_model",path_dir_model)
    # path_file_model = os.path.join(path_dir_model, file_model)
    path_file_model = path_dir_model + '/' + file_model
    # print("path_file_model", path_file_model)

    # check whether the files exist
    # 检查预测文件、原值文件是否存在
    list_path = [path_file_model]
    # print("list_path",list_path)
    check_path_exist(list_path)
    # 导入模型
    model = keras.models.load_model(path_file_model)
    return model

def check_pred_digit(pred, N_digit_pred):
    """
    对输入的pred数组，给出建议的有效位数（以及小数点后位数）
    """
    n_pred_after_dot = 0
    n_pred_before_dot = 0
    for data in pred.flatten():
        a = str(data).split(".")
        n_pred_before_dot = max(n_pred_before_dot, len(a[0]))
        if len(a) == 2:
            n_pred_after_dot = max(n_pred_after_dot, len(a[1]))
    # 如果小数点后数字太多，则降低它
    if n_pred_before_dot + n_pred_after_dot > N_digit_pred:
        if n_pred_before_dot < N_digit_pred:
            n_pred_after_dot = N_digit_pred - n_pred_before_dot
            n_pred = N_digit_pred + 1
        else:
            n_pred_after_dot = 0
            n_pred = n_pred_before_dot
    else:
        n_pred = n_pred_before_dot + n_pred_after_dot
    return n_pred, n_pred_after_dot

def save_pred2txt(dir_pred, pred, dir_model, file_model, W_fmt_out, N_digit_pred, train_name):
    """
    保存预测文件到txt文件
    """
    path_dir_pred = os.path.join(os.curdir, dir_pred)
    # print("path_dir_pred",path_dir_pred) # .\./../prediction
    os.makedirs(path_dir_pred, exist_ok=True)

    file_prefix = 'au150_DW_ann_' + train_name + '_cr1_200_20_1'
    # print("file_prefix",file_prefix)

    file_pred_txt = file_prefix + '_pred.txt'
    path_file_pred_txt = os.path.join(path_dir_pred, file_pred_txt)
    fout = open(path_file_pred_txt,'w')
    if W_fmt_out:
        n_pred, n_pred_after_dot = check_pred_digit(pred, N_digit_pred)
        for i in range(pred.size):
            # print(f"{pred[i][0]:>{n_pred}.{n_pred_after_dot}f}")
            print(f"{pred[i][0]:>{n_pred}.{n_pred_after_dot}f}", file=fout)
    fout.close()

def dict2txt(filepath, dic, note='', wa='w'):
    fout = open(filepath,wa)
    # print(note)
    print(note,file=fout)
    for key in dic.keys():
        # print(f"  {key:<25s}: {dic[key]:>10.7f}")
        print(f"  {key:<25s}: {dic[key]:>10.7f}",file=fout)
    fout.close()

def brief_pred_ana2txt(dir_pred, pred, dir_model, file_model, train_name):
    """
    对预测进行简单分析并输出到txt
    """
    dict_pred = {
            'Pred_maximum'            : np.max(pred),
            'Pred_minimum'            : np.min(pred),
            'Pred_mean'               : np.mean(pred),
            'Pred_variance'           : np.var(pred),
            'Pred_standard deviation' : np.std(pred),
            'Pred_median'             : np.median(pred)
                 }
    path_dir_pred = os.path.join(os.curdir, dir_pred)
    os.makedirs(path_dir_pred, exist_ok=True)

    file_prefix = 'au150_DW_ann_' + train_name + '_cr1_200_20_1'

    file_stat_txt = file_prefix + '_pred_statistics.txt'
    path_file_stat = os.path.join(path_dir_pred, file_stat_txt)
    dict2txt(path_file_stat, dict_pred, "\nPrediction Info:")


