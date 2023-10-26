import math
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def check_path_exist(list_path, W_igore = False):
    """
    检查文件是否存在
    Input  : list of paths or a path
    W_igore : whether to ignore the warning message
              True: ignore the warning message and continue running the code
              False: giving the error message and stop running the code
    Output : None or Error message following the exit of code(W_igore = false) or continue the program(W_igore = True)
    """

    type_input = type(list_path)
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
    else:
        exist = os.path.exists(list_path)
        # print("list_path",list_path)
        if not exist:
            print()
            print(f"** Error!! cannot find file {list_path}! **")
            print()
            if W_igore:
                pass
            else:
                sys.exit()

def read_dict(dict):
    """
    read data from dictionary in dict_file, and check whether the file exsits
    """

    label = dict['label']
    dir = dict['dir']
    file = dict['file']
    # print("dir",dir)
    # print("file",file)
    ### NOTE 此处相对路径加上绝对路径的符号错误 path_file ./../datasets\Au_chi.txt
    # path_file = os.path.join(dir, file)
    path_file = dir + "/" + file
    # print("path_file",path_file)
    check_path_exist(path_file, W_igore = False)

    if label == 'feature':
        tag = None
        data_type = None
        step = None
        pass
    elif label =='label':
        tag = dict['tag']
        data_type = dict['type']
        step = dict['step']

    return label, path_file, file, tag, data_type, step

def digit_dot(step, before_dot = False):
    """
    calculate how many digits after/before dot for the input value.
    """

    step_split = str(step).split(".")
    if before_dot:
        digit = len(step_split[0])
    else:
        if len(step_split) == 2:
            digit = len(step_split[1])
        else:
            digit = 0
    return digit

# 在间隔为step的网格上进行补0的字典补足。
def dict_expand(dict0, dc, step, type='all'):
    """
    Expand the dictionary 'dict0' to the one with keys located at the girds with distance 'step'
    and the corresponding value of keys are 0 for dc = 'c'(continuous);
    wile for dc = 'd'(dispersed), the keys of expanded dictionary will be located in the
    most sparse linear grid which covers all the keys in dict0.
    Input :
        dict0   :   dictionary to be treat
                    dictionary
        dc      :   property of label(key in dict0)
                    String
                    'c' for continuous
                    'd' for dispersed
                    else for wrong case
        step    :   the interval for the two adjacent label indices
                    numeral
        type    :   type for dictionary expansion
                    String
                    'all'  : Expand to all grids
                    'adj0' : Expand to grids only adjacent to non-zero grids
    """

    trivial = 0.0001

    a = list(dict0.keys())
    # print(f"dict0={dict0}")
    # print(f"dict0.keys()={dict0.keys()}")
    # print(f"list(dict0.keys())={a}")
    a.sort()

    new = {}
    if dc == 'd':
        dist = []
        for i in range(len(a)-1):
            dist.append(a[i+1]-a[i])
        dist.sort()
        step = dist[0]

    digit = digit_dot(step)

    if dc == 'c' or dc == 'd':
        b = (a[-1] - a[0])/step
        if abs(b - int(b+trivial)) > trivial:
            print()
            print(f"** Error!! The max and min of keys in dictionary '{dict0}' don't satisfy the definition of step = {step}!")

        n_step = int(b+trivial) + 1
        n_len = len(a)
        if n_len < n_step:
            for c in np.arange(a[0], a[-1], step):

                c = round(c,digit)
                if c not in a:
                    if type == 'all':
                        new[c] = 0
                    elif type == 'adj0':
                        c1 = c - step
                        if c1 in a:
                            new[c] = 0
                    else:
                        pass
        elif n_len == n_step:
            pass
        else:
            print()
            print(f"** Error!! N of items in dictionary '{dict0}' outranges the space that its key allows!")
    else:
        print()
        print(f"** Error!! label type is {dc}, Not continuous nor Dispersed!")
    new.update(dict0)
    return new

# 将字典按照key的顺序排序，返回key和对应于key的value列表
def sort_dict2lists (dict0,reverse = False):
    """
    sort the input dictionary 'dict0', return two lists with sorted keys and the corresponding values.
    """
    list_key = sorted(list(dict0.keys()), reverse=reverse)
    # print(f"list_key={list_key}")
    list_value = []
    for key in list_key:
        list_value.append(dict0[key])
    return list_key, list_value

def save_fig(IMAGES_PATH, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    os.makedirs(IMAGES_PATH, exist_ok=True)
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    # print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    plt.close()

# 绘制各个cluster的CR统计图
def plot_cr(x, y, path, file, delta, xlabel='Coordinate bond length($\AA$)', ylabel='Number', title='CR statistics'):
    """
    绘制CR统计图。
    x：平均配位键长列表
    y: x对应的样本数
    digit：自然数，用于给出绘制图中x轴数字的有效位数，如果步长为0.1，则为1，若步长为0.05， 则为2
    path：图的保存路径
    file：图的名称（不包含后缀）
    """

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)

    # n0, index for plot show.
    n0 = 13
    n1 = 26

    x_index = np.array(x)
    y_index = np.array(y)
    y_max = max(y_index)

    len_x = len(x)

    digit = digit_dot(delta)

    ind = np.linspace(1,len_x,len_x)

    if len_x <= n0:
        fontsize = 10
        rotation = 0
    elif len_x <= n1:
        fontsize = 8
        rotation = 45
    else:
        fontsize = 5
        rotation = 45

    bar_width = 1 # 定义一个数字代表每个独立柱的宽度

    rects1 = ax1.bar(ind, y_index, width=bar_width,align='edge',alpha=0.5, color='yellow',edgecolor='red',label=file)
    for a,b in zip(ind,y_index):
        plt.text(a+0.5, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=fontsize)

    plt.legend() # 显示图例
    plt.tight_layout() # 自动控制图像外部边缘，此方法不能够很好的控制图像间的间隔

    ax1.set_xticks(np.append(ind,len_x+1))
    xx = list(np.append(x_index,round(x[-1]+delta,digit)))
    ax1.set_xticklabels(xx,rotation=rotation,fontsize=fontsize)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.set_ylim(0,y_max*1.2)

    save_fig(path,file)
    plt.close(fig)

# 绘制各个cluster的CN统计图
def plot_cn(x, y, path, file, xlabel='Coordinate number', ylabel='Number', title='CN statistics'):
    """
    绘制CN统计图。
    x：平均配位数列表
    y: x对应的样本数
    path：图的保存路径
    file：图的名称（不包含后缀）
    """

    #fig = plt.figure(num=1, figsize=(15, 8),dpi=80) #开启一个窗口，同时设置大小，分辨率
    fig = plt.figure(1)
    ax1 = fig.add_subplot(1,1,1)

    # n0, index for plot show.
    n0 = 15
    n1 = 30

    x_index = np.array(x)
    y_index = np.array(y)
    y_max = max(y_index)

    len_x = len(x)
    ind = np.linspace(1,len_x,len_x)

    if len_x <= n0:
        fontsize = 10
        rotation = 0
    elif len_x <= n1:
        fontsize = 8
        rotation = 45
    else:
        fontsize = 5
        rotation = 45

    bar_width = 0.8 #定义一个数字代表每个独立柱的宽度

    rects1 = ax1.bar(ind, y_index, width=bar_width,alpha=0.7, color='blue',label=file)

    for a,b in zip(ind,y_index):
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=fontsize)

    plt.legend()
    plt.tight_layout() #自动控制图像外部边缘，此方法不能够很好的控制图像间的间隔

    ax1.set_xticks(ind)
    ax1.set_xticklabels(x,rotation=rotation,fontsize=fontsize)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.set_ylim(0,y_max*1.2)

    save_fig(path,file)
    plt.close(fig)

def plot_label(dict_label_count, dict_label_stat_info, path, file):
    """
    绘制dict_label_count中统计的label值分布
    input:
        label                 :   label to be counted
                                  String
        dict_label_count      :   dictionary to be ploted
                                  Dictionary
        dict_label_stat_info  :   dictionary for label 'label'
                              :   Dictionary
        path                  :   the directory path of plot-figure file
                                  String
        file                  :   the name of plot-figure, without suffix
                                  String
    Return  :   figure file in 'path' with name 'file'

    """
    x, y = sort_dict2lists(dict_label_count)
    dc = dict_label_stat_info['type']
    step = dict_label_stat_info['step']

    if dc == 'c':
        #        plot_cr(x, y, path, file, step, xlabel=xlabel, ylabel=ylabel, title=title)
        plot_cr(x, y, path, file, step)
    elif dc == 'd':
        #        plot_cn(x, y, path, file, xlabel=xlabel, ylabel=ylabel, title=title)
        plot_cn(x, y, path, file)
    else:
        # print()
        print(f"** Error!! label type is {dc}, Not continuous nor Dispersed! 2")

def label_count_plot(file_pre, dict_label_stat_info,
                      dict_label_count,
                     path_work):
    """
    绘制统计的label
    """
    trivial = 0.00001

    dc = dict_label_stat_info['type']
    if dc == 'd' or dc == 'D':
        dc = 'd'
        step = trivial
    elif dc == 'c' or dc == 'C':
        dc = 'c'
        step = dict_label_stat_info['step']
        if step <= 0:
            # print()
            print(f"** step = {step} <= 0, WRONG!")
            sys.exit()
    else:
        # print()
        print(f"** Error!! label type is {dc}, Not continuous nor Dispersed! 5")

    dict_label_count_exp_adj0 = dict_expand(dict_label_count, dc, step, type='adj0')

    if dc == 'd':
        file_plot = file_pre + f"_bar"
    elif dc == 'c':
        file_plot = file_pre + f"_bar_step={step}"

    plot_label(dict_label_count_exp_adj0, dict_label_stat_info, path_work, file_plot)

    return

def locate(data, step, start = False, start0 = 0):
    """
    locate the input data 'data' to the point in the grid with step 'step' if start = True,
    the start point will be start0
    将网格区域-->网格点
    """

    trivial = min(0.00001, step/100)
    digit = digit_dot(step)
    if start:
        n = math.floor((data - start0 + trivial) / step)
        data0 = round(n * step + start0, digit)
    else:
        n = math.floor((data + trivial) / step)
        data0 = round(n * step, digit)

    return data0

def label_count(dataset, dict_label_stat_info):
    """
    统计label
    """
    trivial = 0.00001

    dc = dict_label_stat_info['type']
    if dc == 'd' or dc == 'D':
        dc = 'd'
        step = trivial
    elif dc == 'c' or dc == 'C':
        dc = 'c'
        step = dict_label_stat_info['step']
        if step <= 0:
            # print()
            print(f"** step = {step} <= 0, WRONG!")
            sys.exit()
    else:
        # print()
        print(f"** Error!! label type is {dc}, Not continuous nor Dispersed! 4")

    digit = digit_dot(step)

    dict_label_count = {}
    for i, data in enumerate(dataset.flatten()):
        if dc == 'd':
            a = data
        elif dc == 'c':
            a = locate(data, step)
        else:
            pass

        if a in dict_label_count.keys():
            dict_label_count[a] += 1
        else:
            dict_label_count[a] = 1

    return dict_label_count

