import os
import sys
assert sys.version_info >= (3, 5)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def check_path_exist(list_path, W_igore = False):
    type_input = type(list_path).__name__
    # print(f"type {list_path} = {type_input}")
    if type_input == 'list':
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
    elif type_input == 'str':
        exist = os.path.exists(list_path)
        if not exist:
            print()
            print(f"** Error!! cannot find file {list_path}! **")
            print()
            if W_igore:
                pass
            else:
                sys.exit()
    else:
        print()
        print(f"** Error!! {list_path} is not a path! **")
        print()
        if W_igore:
            pass
        else:
            sys.exit()

### NOTE 修改了两个形参名
def read_label_pred(path_file_label, path_file_pred):
    """
    read label and pred files, output label and pred values (numpy array)
    input:
        path_label : the path of label file
        path_pred  : the path of prediction file
    return:
        label : label values (numpy array)
        pred  : prediction values (numpy array)
    """
    # 读取label文件
    data_df = pd.read_csv(path_file_label, header=None, delim_whitespace=True)
    label = data_df.values
    # 读取pred文件
    data_df = pd.read_csv(path_file_pred, header=None, delim_whitespace=True)
    pred = data_df.values

    # 检查label、pred数据集维度是否相同
    # print("label.shape",label.shape)
    # print("pred.shape",pred.shape)
    if label.size != pred.size:
        print(f"*** Error! size of label {label.size} != size of pred {pred.size}")
        sys.exit()

    return label, pred

def label_ana(label):
    label_mean = np.mean(label)
    label_diff = label - label_mean
    label_diff_abs = np.abs(label_diff)
    label_err = label_diff / label
    label_err_abs = np.abs(label_err)
    label_err_2 = label_diff / np.std(label)
    label_err_2_abs = np.abs(label_err_2)

    list_dict_label_attr = [
        {'N. of Data'     : label.size},
        {'Max'            : np.max(label)},
        {'Min'            : np.min(label)},
        {'mean'           : np.mean(label)},
        {'median'         : np.median(label)},
        {'diff_Max'       : np.max(label_diff)},
        {'diff_Min'       : np.min(label_diff)},
        {'diff_mean'      : np.mean(label_diff)},
        {'diff_median'    : np.median(label_diff)},
        {'|diff|_Max'     : np.max(label_diff_abs)},
        {'|diff|_Min'     : np.min(label_diff_abs)},
        {'|diff|_mean'    : np.mean(label_diff_abs)},
        {'|diff|_median'  : np.median(label_diff_abs)},
        {'Var'            : np.var(label)},
        {'Std'            : np.std(label)},
        {'error_Max'      : np.max(label_err)},
        {'error_Min'      : np.min(label_err)},
        {'error_mean'     : np.mean(label_err)},
        {'error_median'   : np.mean(label_err)},
        {'|error|_max'    : np.max(label_err_abs)},
        {'|error|_min'    : np.min(label_err_abs)},
        {'|error|_mean'   : np.mean(label_err_abs)},
        {'|error|_median' : np.median(label_err_abs)},
        {'error_2_Max'      : np.max(label_err_2)},
        {'error_2_Min'      : np.min(label_err_2)},
        {'error_2_mean'     : np.mean(label_err_2)},
        {'error_2_median'   : np.mean(label_err_2)},
        {'|error_2|_max'    : np.max(label_err_2_abs)},
        {'|error_2|_min'    : np.min(label_err_2_abs)},
        {'|error_2|_mean'   : np.mean(label_err_2_abs)},
        {'|error_2|_median' : np.median(label_err_2_abs)}
    ]
    return list_dict_label_attr

def dict2txt(file, list_dict_predVslabel, list_dict_label_attr):
    """
    将pred对初步统计输出并与label对照
    """
    fout = open(file, 'w')
    str1 = 'Prediction:'
    str2 = 'Label:'
    len_str1 = len(str1)
    # len_str2 = len(str2)

    len_1 = 0
    len_2 = 0
    for data1, data2 in zip(list_dict_predVslabel, list_dict_label_attr):
        for key in data1.keys():
            if key == 'fmt':
                fmt = data1[key]
            else:
                a = key
        len_1 = max(len_1, len(a))

    a = max(len_str1, len_1)

    ### NOTE 加上这一行是个空白行所以注释掉了
    # print(f"\n{str1:<{a}s}{' '*(1+3+10+1)}|{' '*3}{str2}", file=fout)
    for data1, data2 in zip(list_dict_predVslabel, list_dict_label_attr):
        for key in data1.keys():
            if key == 'fmt':
                fmt = data1[key]
            else:
                a = key
        for key in data2.keys():
            b = key
        # print(f" {a:<{len_1}s} : {data1[a]:>{fmt}} | {data2[b]:>{fmt}} : {b:<{len_2}s}")
        print(f" {a:<{len_1}s} : {data1[a]:>{fmt}} | {data2[b]:>{fmt}} : {b:<{len_2}s}",file=fout)
    fout.close()

def pred_ana2csv(label, pred, diff, diff_abs, err, err_abs, err_2, err_2_abs, file):
    df = pd.DataFrame([label.flatten(),
                       pred.flatten(),   # pred.T[0] also works
                       diff.flatten(),
                       diff_abs.flatten(),
                       err.flatten(),
                       err_abs.flatten(),
                       err_2.flatten(),
                       err_2_abs.flatten()])
    df = df.transpose()
    head = [f'Label',
            f'Prediction',
            f'Diff',
            f'|Diff|',
            f'Error',
            f'|Error|',
            f'Error_2',
            f'|Error_2|']
    df.to_csv(file, index=False, header=head)

def save_fig(IMAGES_PATH, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    os.makedirs(IMAGES_PATH, exist_ok=True)
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    # print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    plt.close()