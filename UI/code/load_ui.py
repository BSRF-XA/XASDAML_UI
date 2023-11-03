import json
import sys

import keras

assert sys.version_info >= (3, 5)
import time

import tensorflow as tf

from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QTextOption,QPixmap,QTextCursor,QFont

from util_plot_label import *
from util_nn_train import *
from util_predict import *
from util_predict_analysis import *

# 模型修改窗口
class ModelDialog(QDialog):
    def __init__(self,signal):
        super().__init__()
        self.setWindowTitle("模型结构")
        self.modify_model_signal = signal

        # 创建垂直布局
        layout = QVBoxLayout()

        # 创建输入框和标签
        self.input_num = QLineEdit()
        self.label = QLabel("请输入层数:")

        # 将输入框和标签添加到布局中
        layout.addWidget(self.label)
        layout.addWidget(self.input_num)

        # 设置布局
        self.setLayout(layout)

        # 连接信号和槽
        self.input_num.textChanged.connect(self.on_input_num_changed)

    # @pyqtSlot(str)
    def on_input_num_changed(self, text):
        # 清空布局中的所有控件
        while self.layout().count() > 2:
            item = self.layout().takeAt(2)
            if item.widget():
                item.widget().deleteLater()

        # 根据输入的数动态生成相应数量的输入框
        try:
            self.layer = int(text)

            # 创建一个空字典来存储标签名
            self.variables = {}

            for i in range(self.layer):
                # 使用字符串作为键，动态创建变量名
                var_name = "input_" + str(i)
                self.variables[var_name] = QLineEdit()

                label = QLabel(f"第{i + 1}层节点数:")
                self.layout().addWidget(label)
                self.layout().addWidget(self.variables[var_name])
        except ValueError:
            pass

        btn_layout = QHBoxLayout()
        self.yes_btn = QPushButton("确定")
        self.no_btn = QPushButton("取消")

        self.yes_btn.clicked.connect(self.accept_info)
        self.no_btn.clicked.connect(self.reject_info)

        btn_layout.addWidget(self.yes_btn)
        btn_layout.addWidget(self.no_btn)
        self.layout().addLayout(btn_layout)

    def accept_info(self):
        # 节点数列表
        node_list = []
        # 判断为空或不是数字，弹出提示 节点数不能为空 节点数必须是数字
        for i in range(self.layer):
            if self.variables["input_" + str(i)].text() == "":
                QMessageBox.warning(self, "Error", "节点数不能为空")
            elif not self.variables["input_" + str(i)].text().isdigit():
                QMessageBox.warning(self, "警告", "节点数必须是数字", QMessageBox.Yes)
            else:
                node_list.append(int(self.variables["input_" + str(i)].text()))

        if self.layer == len(node_list):
            model_info_dict = {"layer": self.layer, "node_list": node_list}
            self.modify_model_signal.emit(json.dumps(model_info_dict))
            self.accept()

    def reject_info(self):
        self.accept()

# 图片展示窗口
class ImageDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("结果展示")

        self.layout = QVBoxLayout()
        self.label = QLabel()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

    def set_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.label.setPixmap(pixmap)

# 将控制台信息重定向至前台plainTextEdit进行显示
class EmittingStr(QObject):
    textWritten = pyqtSignal(str)  # 定义一个发送str的信号

    def write(self, text):
        self.textWritten.emit(str(text))
        # 使训练信息在plainTextEdit停留
        loop = QEventLoop()
        QTimer.singleShot(1, loop.quit)
        loop.exec_()

    # 实现实时刷新功能
    def flush(self):  # real signature unknown; restored from __doc__
        """ flush(self) """
        pass

# 子线程
class AnalysisThread(QThread):
    # 创建自定义信号
    get_combobox_signal = pyqtSignal(str)

    def __init__(self,signal):
        super().__init__()
        self.to_plainEdit_signal = signal

    def run(self):
        # print("AnalysisThread线程在执行...")

        file_name = self.path.split('/')[-1]
        # print("file_name",file_name) # Au_chi.txt

        file_path = self.path
        # print("file_path",file_path) # E:/Code/XASDAML/UI/datasets/Au_chi.txt
        exist = os.path.exists(file_path)
        if not exist:
            print(f"** Error!! cannot find file {file_path}! **")

        # 找到对应文件后，对数据进行分析，分析之后写入对应的txt文件，然后再读取,传给主线程让其在结果面板显示

        # 读取源文件
        data_df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        data_np_array = data_df.values
        # print("data_np_array",data_np_array)

        # 目的文件夹
        path_dir_attr = './../datasets_analysis'
        # 生成文件夹
        os.makedirs(path_dir_attr, exist_ok=True)
        # 目的文件名
        file_attr_txt = file_name.split('.')[0] + '_attributes.txt'
        # 目的文件路径
        path_attr_txt = os.path.join(path_dir_attr, file_attr_txt)
        # print(path_attr_txt)

        # data attributes
        attr_dict1 = {
            'file': file_name,
            'data shape': data_np_array.shape,
            'number of dimension': data_np_array.ndim,
            'data number': data_np_array.size,
            'data memory (bytes)': data_np_array.nbytes,
            'data element memory (bytes)': data_np_array.itemsize,
            'data type': data_np_array.dtype
        }
        fout = open(path_attr_txt, 'wt')

        for key in attr_dict1.keys():
            print(f"  {key:<33}: {attr_dict1[key]}", file=fout)

        # data properties (max, min, mean, variance, std)
        # ==============================
        # axis =0表示按行方向计算，即每列求值；1表示按列方向计算，即每行求值
        # 方差函数var()相当于函数mean(abs(x - x.mean())**2),其中x为矩阵；
        # 标准方差std()相当于sqrt(mean(abs(x - x.mean())**2))，或相当于sqrt(x.var())。
        # 中值指的是将序列按大小顺序排列后，排在中间的那个值，如果有偶数个数，则是排在中间两个数的平均值。
        data_mean = np.mean(data_np_array, axis=0)
        delta = data_np_array - data_mean
        delta_abs = np.abs(delta)
        error = delta / data_np_array
        error_abs = np.abs(error)

        attr_dict2 = {
            'Max': np.max(data_np_array, axis=0),
            'Min': np.min(data_np_array, axis=0),
            'Mean': np.mean(data_np_array, axis=0),
            'Median': np.median(data_np_array, axis=0),
            'Delta_max': np.max(delta, axis=0),
            'Delta_min': np.min(delta, axis=0),
            'Delta_mean': np.mean(delta, axis=0),
            'Delta_median': np.median(delta, axis=0),
            '|Delta|_max': np.max(delta_abs, axis=0),
            '|Delta|_min': np.min(delta_abs, axis=0),
            '|Delta|_mean': np.mean(delta_abs, axis=0),
            '|Delta|_median': np.median(delta_abs, axis=0),
            'Var': np.var(data_np_array, axis=0),
            'Std': np.std(data_np_array, axis=0),
            'Error_max': np.max(error, axis=0),
            'Error_min': np.min(error, axis=0),
            'Error_mean': np.mean(error, axis=0),
            'Error_median': np.median(error, axis=0),
            '|Error|_max': np.max(error_abs, axis=0),
            '|Error|_min': np.min(error_abs, axis=0),
            '|Error|_mean': np.mean(error_abs, axis=0),
            '|Error|_median': np.median(error_abs, axis=0),
        }

        print(file=fout)
        for key in attr_dict2.keys():
            fout.write(f"  {key:<15}:")
            for j in range(data_np_array.shape[1]):
                fout.write(f"{attr_dict2[key][j]:14.6e}")
            fout.write('\n')
        fout.close()

        with open(path_attr_txt, "r") as file:
            data = file.read()
        # print(data)

        # 传给主线程
        self.to_plainEdit_signal.emit(data)

    def get_data(self,data):
        self.path = data
        # print(data)

class PlotLabelThread(QThread):
    # 创建自定义信号
    get_combobox_signal = pyqtSignal(str)

    def __init__(self,signal):
        super().__init__()
        self.plot_path_signal = signal

    def label_count_writeout(self, file, dict_label_stat_info,
                             dict_label_count):
        """
        label statistics write to file
        """
        data_type = dict_label_stat_info['type']
        # set the output format of data
        list_label, list_number = sort_dict2lists(dict_label_count)
        n_lab_before_dot = 0
        n_lab_after_dot = 0
        for label in list_label:
            a = str(label).split(".")
            n_lab_before_dot = max(n_lab_before_dot, len(a[0]))
            if len(a) == 2:
                n_lab_after_dot = max(n_lab_after_dot, len(a[1]))
        n_lab_self = n_lab_before_dot + 1 + n_lab_after_dot
        if n_lab_after_dot == 0:
            n_lab_self = n_lab_before_dot

        if data_type == 'c':
            step = dict_label_stat_info['step']
            n_lab_after_dot = max(n_lab_after_dot, digit_dot(step))

        n_number = 0
        for label in list_number:
            n_number = max(n_number, len(str(label)))

        fout = open(file, 'a')
        print(f"\n{self.data} group in sets:", file=fout)
        #
        if data_type == 'd':
            for label, number in zip(list_label, list_number):
                if n_lab_after_dot > 0:
                    print(f"  {label:>{n_lab_self}.{n_lab_after_dot}f} : {number:>{n_number}d}", file=fout)
                else:
                    print(f"  {label:>{n_lab_self}d} : {number:>{n_number}d}", file=fout)
        elif data_type == 'c':
            for label, number in zip(list_label, list_number):
                if n_lab_after_dot > 0:
                    print(
                        f"  {label:>{n_lab_self}.{n_lab_after_dot}f} - {label + step:>{n_lab_self}.{n_lab_after_dot}f} : {number:>{n_number}d}",
                        file=fout)
                else:
                    print(f"  {label:>{n_lab_self}d} : {number:>{n_number}d}", file=fout)
        else:
            pass

        fout.close()

    def run(self):
        # print("PlotThread线程在执行...")

        # print(self.data)  # Au_cr1

        file_name = self.path.split('/')[-1]
        # print("file_name",file_name)

        self.data = file_name.split('.')[0]
        # print(self.data)

        file_path = self.path
        # print("file_path",file_path)
        exist = os.path.exists(file_path)
        if not exist:
            print(f"** Error!! cannot find file {file_path}! **")

        # 找到对应文件后，对数据进行分析画图

        ### 生成txt
        # 读取源文件
        data_df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        data_np_array = data_df.values
        # print("data_np_array",data_np_array)

        # 目的文件夹
        path_dir_attr = './../datasets_analysis'
        # 目的文件名
        file_attr_txt = self.data + '_attributes.txt'
        # 目的文件路径
        path_attr_txt = os.path.join(path_dir_attr, file_attr_txt)
        # print(path_attr_txt)

        # data attributes
        attr_dict1 = {
            'file': file_name,
            'data shape': data_np_array.shape,
            'number of dimension': data_np_array.ndim,
            'data number': data_np_array.size,
            'data memory (bytes)': data_np_array.nbytes,
            'data element memory (bytes)': data_np_array.itemsize,
            'data type': data_np_array.dtype
        }
        fout = open(path_attr_txt, 'wt')

        for key in attr_dict1.keys():
            print(f"  {key:<33}: {attr_dict1[key]}", file=fout)

        # data properties (max, min, mean, variance, std)
        # ==============================
        # axis =0表示按行方向计算，即每列求值；1表示按列方向计算，即每行求值
        # 方差函数var()相当于函数mean(abs(x - x.mean())**2),其中x为矩阵；
        # 标准方差std()相当于sqrt(mean(abs(x - x.mean())**2))，或相当于sqrt(x.var())。
        # 中值指的是将序列按大小顺序排列后，排在中间的那个值，如果有偶数个数，则是排在中间两个数的平均值。
        data_mean = np.mean(data_np_array, axis=0)
        delta = data_np_array - data_mean
        delta_abs = np.abs(delta)
        error = delta / data_np_array
        error_abs = np.abs(error)

        attr_dict2 = {
            'Max': np.max(data_np_array, axis=0),
            'Min': np.min(data_np_array, axis=0),
            'Mean': np.mean(data_np_array, axis=0),
            'Median': np.median(data_np_array, axis=0),
            'Delta_max': np.max(delta, axis=0),
            'Delta_min': np.min(delta, axis=0),
            'Delta_mean': np.mean(delta, axis=0),
            'Delta_median': np.median(delta, axis=0),
            '|Delta|_max': np.max(delta_abs, axis=0),
            '|Delta|_min': np.min(delta_abs, axis=0),
            '|Delta|_mean': np.mean(delta_abs, axis=0),
            '|Delta|_median': np.median(delta_abs, axis=0),
            'Var': np.var(data_np_array, axis=0),
            'Std': np.std(data_np_array, axis=0),
            'Error_max': np.max(error, axis=0),
            'Error_min': np.min(error, axis=0),
            'Error_mean': np.mean(error, axis=0),
            'Error_median': np.median(error, axis=0),
            '|Error|_max': np.max(error_abs, axis=0),
            '|Error|_min': np.min(error_abs, axis=0),
            '|Error|_mean': np.mean(error_abs, axis=0),
            '|Error|_median': np.median(error_abs, axis=0),
        }

        print(file=fout)
        for key in attr_dict2.keys():
            fout.write(f"  {key:<15}:")
            for j in range(data_np_array.shape[1]):
                fout.write(f"{attr_dict2[key][j]:14.6e}")
            fout.write('\n')
        fout.close()

        ### 保存图片
        # read the file data
        data_df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        data_np_array = data_df.values

        # Set label_stat_info
        if self.data == 'Au_cn1':
            dict_label_stat_info = {'type': 'd', 'step': 0}
        if self.data == 'Au_cr1':
            dict_label_stat_info = {'type': 'c', 'step': 0.05}

        # label count statistics
        dict_label_count = label_count(data_np_array, dict_label_stat_info)

        # label statistics plot
        file_name = os.path.splitext(file_name)
        # print("file_name",file_name) # ('Au_cn1', '.txt')
        file_pre = file_name[0]
        # print("file_pre",file_pre) # file_pre Au_cn1
        label_count_plot(file_pre, dict_label_stat_info,
                         dict_label_count,
                         path_dir_attr)

        # save count data into attribute file
        file_attr_txt = file_name[0] + '_attributes.txt'
        path_attr_txt = os.path.join(path_dir_attr, file_attr_txt)
        self.label_count_writeout(path_attr_txt, dict_label_stat_info, dict_label_count)
        fout.close()

        # 返回生成图片路径
        if self.data == 'Au_cn1':
            png_name = self.data + '_bar.png'
            png_path = os.path.join(path_dir_attr, png_name)
        if self.data == 'Au_cr1':
            png_name = self.data + '_bar_step=0.05.png'
            png_path = os.path.join(path_dir_attr, png_name)

        # # 传给主线程
        self.plot_path_signal.emit(png_path)

    def get_data(self,path):
        self.path = path

class NNTrainThread(QThread):
    # 创建自定义信号
    get_data_signal = pyqtSignal(str)

    def __init__(self,signal):
        super().__init__()
        self.train_end_signal = signal

    def run(self):
        # print("NNTrainThread线程在执行...")

        # 源文件夹
        dir_data = './../datasets'

        # 源文件名
        file_train = [self.data + '_train.txt', 'Au_cr1_train.txt']
        file_valid = [self.data + '_valid.txt', 'Au_cr1_valid.txt']
        file_test = [self.data + '_test.txt', 'Au_cr1_test.txt']

        # 目标文件夹
        dir_output = './../nn_train'
        # print(os.path.abspath(dir_output))
        os.makedirs(dir_output, exist_ok=True)

        self.label_index = 'CR1'

        name = self.data.split("_")[1]
        # print(name) # chi
        self.model_pre = 'au150_DW_ann_' + name + '_cr1_200_20_1'

        # 当前目录的绝对路径
        path = os.path.abspath(os.curdir)

        # 源文件夹的绝对路径
        path_dir_data = os.path.join(path, dir_data)
        # print(path_dir_data)
        # 为了能够尽量少改动代码，在此处将选中的数据集拷贝到创建的datasets文件夹中
        os.makedirs(path_dir_data, exist_ok=True)

        # # 进行文件拷贝文件
        # data_file = self.data_path
        # # print("data_file",data_file)
        # label_file = self.label_path
        #
        # copy_data_name = os.path.basename(data_file)
        # # print(copy_data_name)
        # destination_file1 = os.path.join(path_dir_data, copy_data_name)  # 拼接目标文件的路径
        # # print(destination_file1)
        # shutil.copy2(data_file, destination_file1)
        #
        # copy_label_name = os.path.basename(label_file)
        # # print(copy_label_name)
        # destination_file2 = os.path.join(path_dir_data, copy_label_name)  # 拼接目标文件的路径
        # shutil.copy2(label_file, destination_file2)
        # # print("拷贝成功!")

        dir_log = dir_output + './my_logs'
        # 日志文件夹
        self.root_logdir = os.path.join(os.curdir, dir_log)
        # print(self.root_logdir)

        file_data = file_train + file_valid + file_test
        # print(file_data) # ['Au_chi_train.txt', 'Au_cr1_train.txt', 'Au_chi_valid.txt', 'Au_cr1_valid.txt', 'Au_chi_test.txt', 'Au_cr1_test.txt']

        for i in range(len(file_data)):
            path_file = os.path.join(path_dir_data, file_data[i])
            exist = os.path.exists(path_file)
            if not exist:
                print()
                print(f"** Error!! cannot find file {file_data[i]} in directory {dir_data}! **")
                print()
                sys.exit()

        for i in range(len(file_data)):
            path_file = os.path.join(path_dir_data, file_data[i])
            # print("path_file",path_file) # E:\Code\XASDAML\ML module\code\./../datasets\Au_chi_train.txt
            data_df = pd.read_csv(path_file, header=None, delim_whitespace=True)
            if i == 0: feature_train = data_df.values
            if i == 1: label_train = data_df.values
            if i == 2: feature_valid = data_df.values
            if i == 3: label_valid = data_df.values
            if i == 4: feature_test = data_df.values
            if i == 5: self.label_test = data_df.values

        # print('train:', feature_train.shape, label_train.shape)
        # print('valid:', feature_valid.shape, label_valid.shape)
        # print('test:', feature_test.shape, self.label_test.shape)

        input_shape = feature_train.shape[1:]

        model_time = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        self.model_pre_time = self.model_pre + '_' + model_time
        # print(self.model_pre_time) # au150_DW_ann_chi_cr1_200_20_1_run_2023_09_26-14_24_04

        self.path_plot = os.path.join(dir_output, self.model_pre)
        # print(self.path_plot) # ./../nn_train\au150_DW_ann_chi_cr1_200_20_1
        os.makedirs(self.path_plot, exist_ok=True)

        file_ana_txt = self.model_pre + '_analysis.txt'
        path_ana_txt = os.path.join(dir_output, file_ana_txt)
        # print(path_ana_txt) # ./../nn_train\au150_DW_ann_chi_cr1_200_20_1_analysis.txt

        fout = open(path_ana_txt, 'w')
        # print(f"Model : {self.model_pre}")
        print(f"Model : {self.model_pre}", file=fout)

        # Set random seed to be used as argument for other functions.
        rseed = 42
        activation = 'relu'

        # print(f"===================================")
        print(f"\n===================================", file=fout)
        # print(f"rseed : {rseed}")
        print(f"rseed : {rseed}", file=fout)
        # print(f"Time : {model_time}")
        print(f"Time : {model_time}", file=fout)

        keras.backend.clear_session()
        np.random.seed(rseed)
        tf.random.set_seed(rseed)

        # model = keras.models.Sequential([
        #     keras.layers.Flatten(input_shape=input_shape),
        #     keras.layers.Dense(200, activation=activation),
        #     keras.layers.Dense(20, activation=activation),
        #     keras.layers.Dense(1, activation=activation)
        # ])
        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=input_shape))

        # 通过循环把中间层造出来
        for i in range(self.layer):
            model.add(keras.layers.Dense(self.node[i], activation=activation))

        model.add(keras.layers.Dense(1, activation=activation))

        # model.summary()

        # 生成模型的整体结构图
        model_full_fig = os.path.join(self.path_plot, self.model_pre + '_model_full.png')
        # print(model_full_fig) # ./../nn_train\au150_DW_ann_chi_cr1_200_20_1\au150_DW_ann_chi_cr1_200_20_1_model_full.png

        ### NOTE 此处报错，从电脑端安装了Graphviz
        keras.utils.plot_model(model, model_full_fig, show_shapes=True, dpi=180)

        model.compile(optimizer='rmsprop', loss='mse')

        # 将训练最好的模型保存下来
        # 设定最好模型的名称，便于保存后调用。
        model_saved_best = os.path.join(dir_output, self.model_pre + ".h5")

        # 设置调回和早停点
        checkpoint_cb = keras.callbacks.ModelCheckpoint(model_saved_best, save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=40, restore_best_weights=True)

        run_logdir = self.get_run_logdir()
        # 设置 Tensorboard
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

        self.history = model.fit(feature_train, label_train, epochs=self.epoch, batch_size=32, verbose=2,
                            validation_data=(feature_valid, label_valid),
                            callbacks=[checkpoint_cb
                                , early_stopping_cb
                                , tensorboard_cb])

        # 恢复控制台输出
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        loss_val = model.evaluate(feature_valid, label_valid, verbose=2)
        # print(f"loss_val = {loss_val:>10.4f}")
        print(f"\nloss_val = {loss_val:>10.4f}", file=fout)

        # # 画loss图
        # plot_LearningCurves([0,20],ind=0)
        # plot_LearningCurves([0.1,1],ind=2)
        self.plot_LearningCurves([0, 0.1], ind=0)

        self.pre = model.predict(feature_test, verbose=2)

        # print(f"self.pre.shape : {self.pre.shape}")
        print(f"\nself.pre.shape : {self.pre.shape}", file=fout)

        # 预测结果与标签的差值
        diff = self.pre - self.label_test
        # print("diff",diff)

        # 得到预测值和标签之间的差异diff的分析结果
        a, self.err = pre_ana(self.pre, self.label_test)
        # print("Prediction:")
        print("\nPrediction:", file=fout)
        for key in a.keys():
            # print(f"  {key:<20}: {a[key]:>11.7f}")
            print(f"  {key:<20}: {a[key]:>11.7f}", file=fout)
        fout.close()

        # 预测值分析结果
        file_pre_txt = self.model_pre + '_pre.txt'
        path_pre_txt = os.path.join(dir_output, file_pre_txt)
        # print(path_pre_txt)  # ./../nn_train\au150_DW_ann_chi_cr1_200_20_1_pre.txt

        fout = open(path_pre_txt, 'w')
        str_no = 'No.'
        str_lab = 'Label'
        str_pre = 'Predict'
        str_diff = 'Diff'
        str_err = 'Error'
        # print(f"{str_no:>5} {str_lab:>5} {str_pre:>8} {str_diff:>8} {str_err:>11}")
        print(f"{str_no:>5} {str_lab:>5} {str_pre:>8} {str_diff:>8} {str_err:>11}", file=fout)

        for i in range(self.pre.size):
            # print(f"{i + 1:>5d} {self.label_test[i][0]:>5.1f} {self.pre[i][0]:>8.4f} {diff[i][0]:>8.4f} {self.err[i][0]:>11.3E}")
            print(f"{i + 1:>5d} {self.label_test[i][0]:>5.1f} {self.pre[i][0]:>8.4f} {diff[i][0]:>8.4f} {self.err[i][0]:>11.3E}",
                  file=fout)
        fout.close()

        # 传给主线程
        self.train_end_signal.emit("训练已结束")

    def get_data(self, data_node_json):
        data_node_json = json.loads(data_node_json)
        # 取到的是训练集
        self.data = data_node_json.get("data")
        self.data_path = data_node_json.get("data_path")
        # print(self.data_path)
        self.label_path = data_node_json.get("label_path")
        # print(self.label_path)
        # print(self.data)
        # 取到的是层数
        self.layer = int(data_node_json.get("layer"))
        # print(self.layer)
        # 取到的是节点数
        self.node = data_node_json.get("node")
        # 将列表中的字符串全部转换为int
        self.node = list(map(int, self.node))
        # print(self.node)
        self.epoch = int(data_node_json.get("epoch"))

    # 绘制训练曲线
    def plot_LearningCurves(self, ylim, ind=0):
        learning_curves_plot = self.model_pre + '_learning_curves' + f"_{ind}"
        # print("learning_curves_plot",learning_curves_plot) # au150_DW_ann_chi_cr1_200_20_1_learning_curves_0
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(ylim)
        plt.legend(loc='upper right')
        plt.title(f"{self.model_pre}\nLearning Curves", fontsize=16)
        save_fig(self.path_plot, learning_curves_plot)
        plt.close()

    def get_run_logdir(self):
        model_time = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        run_id = self.model_pre + '_' + model_time
        return os.path.join(self.root_logdir, run_id)

    def plot_PredictionCompare(self, x_low, x_high, x_ticks=10):
        prediction_compare = self.model_pre + f"_prediction_compare_{self.label_index}_{x_low}_{x_high}"
        x = range(self.pre.shape[0])
        plt.figure(figsize=(8, 8))
        plt.scatter(x, self.label_test, marker='o', label=f'{self.label_index}')
        plt.scatter(x, self.pre, marker='^', label=f'{self.label_index}_pre')
        plt.xlabel('Test series')
        plt.ylabel(f'{self.label_index}')
        plt.xlim(x_low, x_high)
        plt.ylim(min(np.min(self.label_test), np.min(self.pre)), max(np.max(self.label_test), np.max(self.pre)))
        plt.xticks(np.arange(x_low, x_high + 1, x_ticks), rotation=0)  # rotation控制倾斜角度
        plt.legend(loc='upper right', fontsize=12)
        plt.title(f"{self.model_pre}\nRegression Performance", fontsize=16)
        save_fig(self.path_plot, prediction_compare)
        plt.close()

class StopTrainThread(QThread):
    def __init__(self,thread):
        super().__init__()
        self.training_thread = thread

    def run(self):
        # 恢复控制台输出
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # print("StopTrainThread线程在执行...")

        if self.training_thread is not None and self.training_thread.isRunning():
            self.training_thread.terminate()  # 终止训练线程

class ShowPreLabelThread(QThread):
    # 创建自定义信号
    get_combobox_signal = pyqtSignal(str)

    def __init__(self,signal):
        super().__init__()
        self.to_plainEdit_signal = signal

    def run(self):
        # print("ShowPreLabelThread线程在执行...")
        # print(self.data)

        file_name = 'au150_DW_ann_' + self.data + '_cr1_200_20_1_pre.txt'
        # 源文件夹
        dir_data = './../nn_train'
        file_path = os.path.join(dir_data, file_name)
        # print(file_path) # ./../nn_train\au150_DW_ann_chi_cr1_200_20_1_pre.txt

        exist = os.path.exists(file_path)
        if not exist:
            print(f"** Error!! cannot find file {file_path}! **")

        with open(file_path, "r") as file:
            data = file.read()
        # print(data)

        # 传给主线程
        self.to_plainEdit_signal.emit(data)

    def get_data(self,data):
        self.data = data

class ShowPreAnalysisThread(QThread):
    # 创建自定义信号
    get_combobox_signal = pyqtSignal(str)

    def __init__(self, signal):
        super().__init__()
        self.to_plainEdit_signal = signal

    def run(self):
        # print("ShowPreAnalysisThread线程在执行...")

        file_name = 'au150_DW_ann_' + self.data + '_cr1_200_20_1_analysis.txt'
        # 源文件夹
        dir_data = './../nn_train'
        file_path = os.path.join(dir_data, file_name)

        exist = os.path.exists(file_path)
        if not exist:
            print(f"** Error!! cannot find file {file_path}! **")

        with open(file_path, "r") as file:
            data = file.read()
        # print(data)

        # 传给主线程
        self.to_plainEdit_signal.emit(data)

    def get_data(self, data):
        self.data = data

class ShowLossThread(QThread):
    # 创建自定义信号
    get_combobox_signal = pyqtSignal(str)

    def __init__(self, signal):
        super().__init__()
        self.plot_path_signal = signal

    def run(self):
        # print("ShowLossThread线程在执行...")

        file_pre = 'au150_DW_ann_' + self.data + '_cr1_200_20_1'

        dir_data = './../nn_train/'
        # 源文件夹
        folder_path = os.path.join(dir_data, file_pre)
        # print(file_path) # ./../nn_train/au150_DW_ann_chi_cr1_200_20_1

        file_name = file_pre + '_learning_curves_0.png'
        png_path = os.path.join(folder_path, file_name)
        # print(file_path) # ./../nn_train/au150_DW_ann_chi_cr1_200_20_1\au150_DW_ann_chi_cr1_200_20_1_learning_curves_0.png

        exist = os.path.exists(png_path)
        if not exist:
            print(f"** Error!! cannot find file {png_path}! **")

        # 传给主线程
        self.plot_path_signal.emit(png_path)

    def get_data(self, data):
        self.data = data

class ShowModelThread(QThread):
    # 创建自定义信号
    get_combobox_signal = pyqtSignal(str)

    def __init__(self, signal):
        super().__init__()
        self.plot_path_signal = signal

    def run(self):
        # print("ShowModelThread线程在执行...")

        file_pre = 'au150_DW_ann_' + self.data + '_cr1_200_20_1'

        dir_data = './../nn_train/'
        # 源文件夹
        folder_path = os.path.join(dir_data, file_pre)
        # print(file_path) # ./../nn_train/au150_DW_ann_chi_cr1_200_20_1

        file_name = file_pre + '_model_full.png'
        png_path = os.path.join(folder_path, file_name)

        exist = os.path.exists(png_path)
        if not exist:
            print(f"** Error!! cannot find file {png_path}! **")

        # 传给主线程
        self.plot_path_signal.emit(png_path)

    def get_data(self, data):
        self.data = data

class PredictThread(QThread):
    # 创建自定义信号
    get_data_signal = pyqtSignal(str)

    def __init__(self, signal):
        super().__init__()
        self.predict_end_signal = signal

    def run(self):
        # print("PredictThread线程在执行...")

        # self.data Au_chi_test
        if self.model == 'current_training_model':
            set_parameters = [{'index': 1
                                  , 'dir_model': './../nn_train'
                                  , 'file_model': 'au150_DW_ann_chi_cr1_200_20_1.h5'
                                  , 'dir_data': './../datasets'
                                  , 'file_feature': self.data + '.txt'
                                  , 'dir_pred': './../prediction'
                                  , 'dict_fmt_out': {'W_fmt_out': True
                    , 'N_digit_pred': 6}
                               }]

            for ith, param in enumerate(set_parameters):
                # 读取用户设置的参数
                self.read_param(param)

                # 数据、模型导入
                feature_test = import_data(dir_data, file_feature)

                model = import_model(dir_model, file_model)
                # 预测
                pred = model.predict(feature_test)

                # 训练集名
                name = self.data.split('_')[1]
                # print("name",name) # chi

                # 保存预测结果
                save_pred2txt(dir_pred, pred, dir_model, file_model, W_fmt_out, N_digit_pred, train_name=name)
                # 简单分析和输出
                brief_pred_ana2txt(dir_pred, pred, dir_model, file_model, train_name=name)

        else:
            name = self.model.split('_')[1]
            # print("name",name) # chi

            set_parameters = [{'index': 1
                                  , 'dir_model': './../train_model'
                                  , 'file_model': 'au150_DW_ann_' + name + '_cr1_200_20_1.h5'
                                  , 'dir_data': './../datasets'
                                  , 'file_feature': self.data + '.txt'
                                  , 'dir_pred': './../prediction'
                                  , 'dict_fmt_out': {'W_fmt_out': True
                    , 'N_digit_pred': 6}
                               }]

            for ith, param in enumerate(set_parameters):
                # 读取用户设置的参数
                self.read_param(param)

                # 数据、模型导入
                feature_test = import_data(dir_data, file_feature)

                model = import_model(dir_model, file_model)
                # 预测
                pred = model.predict(feature_test)

                # 保存预测结果
                save_pred2txt(dir_pred, pred, dir_model, file_model, W_fmt_out, N_digit_pred, train_name=name)
                # 简单分析和输出
                brief_pred_ana2txt(dir_pred, pred, dir_model, file_model, train_name=name)

        # 传给主线程
        self.predict_end_signal.emit("预测已结束")

    def get_data(self, test_model_json):
        test_model_json = json.loads(test_model_json)
        # print(test_model_json)
        # 取到的是测试集
        self.path = test_model_json.get("path")

        self.data = test_model_json.get("data")
        # print(self.data)
        # 取到的是模型
        self.model = test_model_json.get("model")
        # print(self.model)

    def read_param(self, param):
        """
        读取用户设置参数
        """
        global dir_model, file_model, dir_data, file_feature, dir_pred
        global W_fmt_out, N_digit_pred

        dir_model = param['dir_model']
        file_model = param['file_model']
        dir_data = param['dir_data']
        file_feature = param['file_feature']
        dir_pred = param['dir_pred']
        dict_fmt_out = param['dict_fmt_out']
        W_fmt_out = dict_fmt_out['W_fmt_out']
        if W_fmt_out:
            N_digit_pred = dict_fmt_out['N_digit_pred']
        else:
            N_digit_pred = 20

class ShowPredictAnalysisThread(QThread):
    # 创建自定义信号
    get_data_signal = pyqtSignal(str)

    def __init__(self, signal):
        super().__init__()
        self.to_plainEdit_signal = signal

    def run(self):
        # print("ShowPredictAnalysisThread线程在执行...")
        # print(self.data) # Au_chi_test

        self.name = self.data.split('_')[1]
        # print(self.name)
        set_parameters_analysis = [{'index': 1
                                    # 源文件夹
                                       , 'dir_pred': './../prediction'
                                    # 源文件名
                                       , 'file_pred': 'au150_DW_ann_' + self.name + '_cr1_200_20_1_pred.txt'
                                    # 对应label值文件夹
                                       , 'dir_label': './../datasets'
                                    # 对应label值文件名
                                       , 'file_label': 'Au_cr1_test.txt'
                                    # 目的文件夹
                                       , 'dir_ana': './../prediction_analysis'
                                    # 目的文件名前缀
                                       , 'file_pre': 'au150_DW_ann_' + self.name + '_cr1_200_20_1'
                                       , 'feature_index': self.name
                                       , 'label_index': 'CR1'
                                    # label的类型:
                                    # 'd' : 离散的，即值是离散分布的，不是连续的，比如配位数（都是整数）
                                    # 'c' : 连续的，即值是连续分布的，比如键长
                                    # 注意：由于这里采用的都是回归方法，因此预测值（prediction）类型都是连续的（c）
                                       , 'dict_label': {
                'type': 'c'
                , 'step': 0.02
            }

                                    # step:在对label值和预测值统计时，设置的间隔，即多大间隔的label值或预测值会统计在一起
                                    # 注意：如果类型是离散的，则‘step’的值程序不会使用，但用户仍然需要对其进行设置，不能删除！！
                                    # step_err : 用于预测误差分析统计时的间隔设置，一般是0.01（1%），如果预测误差很大，为了绘图方便，可以设置大点，如0.05（5%）等。
                                    # step_err_2 : 用于预测误差(error_2)分析统计时的间隔设置，一般是0.1（10%），如果预测误差很大，为了绘图方便，可以设置大点，如0.3（30%）等。
                                       , 'dict_pred': {
                'step': 0.02
                , 'step_err': 0.01
                , 'step_err_2': 0.1
            }
                                    }
                                   ]
        for ith, param in enumerate(set_parameters_analysis):
            # 读取用户设置的参数
            self.read_param_analysis(param)

            # 标签路径
            path_file_label = os.path.join(os.curdir, dir_label, file_label)
            # print(path_file_label) # .\./../datasets\Au_cr1_test.txt
            # 源文件路径
            path_file_pred = os.path.join(os.curdir, dir_pred, file_pred)
            # print(path_file_pred) # .\./../prediction\au150_DW_ann_chi_cr1_200_20_1_run_2023_09_27-10_29_11_pred.txt

            # 检查预测文件、原值文件是否存在
            list_path = [path_file_label, path_file_pred]
            check_path_exist(list_path)

            # 导入feature和label的数据
            label, pred = read_label_pred(path_file_label, path_file_pred)

            # —————————————————————————————————————————————————————————————————————————————————————————————————
            # 整体比较
            # 常规比较prediction和label的值
            list_dict_predVslabel, diff, diff_abs, err, err_abs, err_2, err_2_abs = self.pred_ana(pred, label)

            # 将比较结果保存为csv、txt文件
            path_dir_ana = os.path.join(os.curdir, dir_ana)
            self.save2csvtxt(path_dir_ana, file_pre, label, pred, diff, diff_abs, err, err_abs, err_2, err_2_abs)

            # 对整体预测进行统计并保存为txt文件
            # 预测分析文件名和路径
            file_ana_txt = file_pre + '_pred_analysis.txt'
            path_file_ana_txt = os.path.join(path_dir_ana, file_ana_txt)

            # label statistics
            list_dict_label_attr = label_ana(label)
            dict2txt(path_file_ana_txt, list_dict_predVslabel, list_dict_label_attr)

        # 读取生成的整体分析
        file_name = 'au150_DW_ann_' + self.name + '_cr1_200_20_1_pred_analysis.txt'
        # 源文件夹
        dir_data = './../prediction_analysis'
        file_path = os.path.join(dir_data, file_name)

        exist = os.path.exists(file_path)
        if not exist:
            print(f"** Error!! cannot find file {file_path}! **")

        with open(file_path, "r") as file:
            data = file.read()

        # 传给主线程
        self.to_plainEdit_signal.emit(data)

    def get_data(self, data):
        self.data = data

    def read_param_analysis(self, param):
        global dir_pred, file_pred, dir_label, file_label, dir_ana, file_pre
        global feature_index, label_index, dict_label, dict_pred

        dir_pred = param['dir_pred']
        file_pred = param['file_pred']
        dir_label = param['dir_label']
        file_label = param['file_label']
        dir_ana = param['dir_ana']
        file_pre = param['file_pre']
        feature_index = param['feature_index']
        label_index = param['label_index']
        dict_label = param['dict_label']
        dict_pred = param['dict_pred']

    def pred_ana(self, pred, label):
        global diff, diff_abs, err, err_abs, err_2, err_2_abs
        global n_lab_self, n_lab_after_dot, n_lab_before_dot
        global n_pred_self, n_pred_after_dot, n_pred_before_dot
        global n_pred_abs_self, n_pred_abs_after_dot, n_pred_abs_before_dot
        global n_diff_self, n_diff_after_dot, n_diff_before_dot
        global n_diff_abs_self, n_diff_abs_after_dot, n_diff_abs_before_dot
        global n_err_self, n_err_after_dot
        global n_err_abs_self, n_err_abs_after_dot
        global n_err_2_self, n_err_2_after_dot
        global n_err_2_abs_self, n_err_2_abs_after_dot

        diff = pred - label
        diff_abs = np.abs(diff)
        diff2 = np.square(diff)  # = diff**2

        pred_var = np.mean(diff2)
        pred_std = np.sqrt(pred_var)  # = pred_var_mean**0.5
        err = diff / label
        err_abs = np.abs(err)

        err_2 = diff / np.std(label)
        err_2_abs = np.abs(err_2)

        n_lab_after_dot = 0
        n_lab_before_dot = 0
        for data in label.flatten():
            a = str(data).split(".")
            n_lab_before_dot = max(n_lab_before_dot, len(a[0]))
            if len(a) == 2:
                n_lab_after_dot = max(n_lab_after_dot, len(a[1]))
        n_lab_self = n_lab_before_dot + 1 + n_lab_after_dot
        if n_lab_after_dot == 0:
            n_lab_self = n_lab_before_dot

        n_pred_after_dot = 0
        n_pred_before_dot = 0
        for data in pred.flatten():
            a = str(data).split(".")
            n_pred_before_dot = max(n_pred_before_dot, len(a[0]))
            if len(a) == 2:
                n_pred_after_dot = max(n_pred_after_dot, len(a[1]))
        n_pred_self = n_pred_before_dot + 1 + n_pred_after_dot
        if n_pred_after_dot == 0:
            n_pred_self = n_pred_before_dot

        n_diff_after_dot = 0
        n_diff_before_dot = 0
        for data in diff.flatten():
            a = str(data).split(".")
            n_diff_before_dot = max(n_diff_before_dot, len(a[0]))
            if len(a) == 2:
                n_diff_after_dot = max(n_diff_after_dot, len(a[1]))
        n_diff_after_dot = min(max(n_diff_after_dot, n_lab_after_dot), n_pred_after_dot)
        n_diff_self = n_diff_before_dot + 1 + n_diff_after_dot
        if n_diff_after_dot == 0:
            n_diff_self = n_diff_before_dot

        n_diff_abs_after_dot = 0
        n_diff_abs_before_dot = 0
        for data in diff_abs.flatten():
            a = str(data).split(".")
            n_diff_abs_before_dot = max(n_diff_abs_before_dot, len(a[0]))
            if len(a) == 2:
                n_diff_abs_after_dot = max(n_diff_abs_after_dot, len(a[1]))
        n_diff_abs_after_dot = min(max(n_diff_abs_after_dot, n_lab_after_dot), n_pred_after_dot)
        n_diff_abs_self = n_diff_abs_before_dot + 1 + n_diff_abs_after_dot
        if n_diff_abs_after_dot == 0:
            n_diff_abs_self = n_diff_abs_before_dot

        # error and |error| will be formatted as -4.326E-02
        n_err_after_dot = 3
        n_err_self = n_err_after_dot + 4 + 2 + 1
        n_err_abs_after_dot = n_err_after_dot
        n_err_abs_self = n_err_after_dot + 4 + 2

        # error_2 and |error_2| will be formatted as -4.326E-02
        n_err_2_after_dot = n_err_after_dot
        n_err_2_self = n_err_2_after_dot + 4 + 2 + 1
        n_err_2_abs_after_dot = n_err_2_after_dot
        n_err_2_abs_self = n_err_2_after_dot + 4 + 2

        # var
        a = str(pred_var).split(".")
        n_var_before_dot = len(a[0])
        if len(a) == 2:
            # n_var_after_dot = len(a[1])
            n_var_after_dot = min(len(a[1]), n_diff_abs_after_dot)
            n_var = n_var_before_dot + 1 + n_var_after_dot
        else:
            n_var_after_dot = 0
        if n_var_after_dot == 0:
            n_var = n_var_before_dot
        # std
        a = str(pred_std).split(".")
        n_std_before_dot = len(a[0])
        if len(a) == 2:
            # n_std_after_dot = len(a[1])
            n_std_after_dot = min(len(a[1]), n_diff_abs_after_dot)
            n_std = n_std_before_dot + 1 + n_std_after_dot
        else:
            n_std_after_dot = 0
        if n_std_after_dot == 0:
            n_std = n_std_before_dot

        n_all = max(n_lab_self, n_pred_self, n_diff_self, n_diff_abs_self, n_err_self, n_err_abs_self, n_err_2_self,
                    n_err_2_abs_self, n_var, n_std)

        list_dict_predVslabel = [
            {'N. of Data': pred.size, 'fmt': f"{n_all}d"},
            {'Pred_Max': np.max(pred), 'fmt': f"{n_all}.{n_pred_after_dot}f"},
            {'Pred_Min': np.min(pred), 'fmt': f"{n_all}.{n_pred_after_dot}f"},
            {'Pred_mean': np.mean(pred), 'fmt': f"{n_all}.{n_pred_after_dot}f"},
            {'Pred_median': np.median(pred), 'fmt': f"{n_all}.{n_pred_after_dot}f"},
            {'diff_Max': np.max(diff), 'fmt': f"{n_all}.{n_diff_after_dot}f"},
            {'diff_Min': np.min(diff), 'fmt': f"{n_all}.{n_diff_after_dot}f"},
            {'diff_mean': np.mean(diff), 'fmt': f"{n_all}.{n_diff_after_dot}f"},
            {'diff_median': np.median(diff), 'fmt': f"{n_all}.{n_diff_after_dot}f"},
            {'|diff|_Max': np.max(diff_abs), 'fmt': f"{n_all}.{n_diff_abs_after_dot}f"},
            {'|diff|_Min': np.min(diff_abs), 'fmt': f"{n_all}.{n_diff_abs_after_dot}f"},
            {'|diff|_mean(MAE)': np.mean(diff_abs), 'fmt': f"{n_all}.{n_diff_abs_after_dot}f"},
            {'|diff|_median': np.median(diff_abs), 'fmt': f"{n_all}.{n_diff_abs_after_dot}f"},
            {'pred_Var(MSE)': pred_var, 'fmt': f"{n_all}.{n_std_after_dot}f"},
            {'pred_Std(RMSE)': pred_std, 'fmt': f"{n_all}.{n_pred_after_dot}f"},
            {'Error_Max': np.max(err), 'fmt': f"{n_all}.{n_err_after_dot}E"},
            {'Error_Min': np.min(err), 'fmt': f"{n_all}.{n_err_after_dot}E"},
            {'Error_mean': np.mean(err), 'fmt': f"{n_all}.{n_err_after_dot}E"},
            {'Error_median': np.median(err), 'fmt': f"{n_all}.{n_err_after_dot}E"},
            {'|Error|_Max': np.max(err_abs), 'fmt': f"{n_all}.{n_err_abs_after_dot}f"},
            {'|Error|_Min': np.min(err_abs), 'fmt': f"{n_all}.{n_err_abs_after_dot}f"},
            {'|Error|_mean(MAPE)': np.mean(err_abs), 'fmt': f"{n_all}.{n_err_abs_after_dot}f"},
            {'|Error|_median': np.median(err_abs), 'fmt': f"{n_all}.{n_err_abs_after_dot}f"},
            {'Error_2_Max': np.max(err_2), 'fmt': f"{n_all}.{n_err_2_after_dot}E"},
            {'Error_2_Min': np.min(err_2), 'fmt': f"{n_all}.{n_err_2_after_dot}E"},
            {'Error_2_mean': np.mean(err_2), 'fmt': f"{n_all}.{n_err_2_after_dot}E"},
            {'Error_2_median': np.median(err_2), 'fmt': f"{n_all}.{n_err_2_after_dot}E"},
            {'|Error_2|_Max': np.max(err_2_abs), 'fmt': f"{n_all}.{n_err_2_abs_after_dot}f"},
            {'|Error_2|_Min': np.min(err_2_abs), 'fmt': f"{n_all}.{n_err_2_abs_after_dot}f"},
            {'|Error_2|_mean(MAPE)': np.mean(err_2_abs), 'fmt': f"{n_all}.{n_err_2_abs_after_dot}f"},
            {'|Error_2|_median': np.median(err_2_abs), 'fmt': f"{n_all}.{n_err_2_abs_after_dot}f"}
        ]
        return list_dict_predVslabel, diff, diff_abs, err, err_abs, err_2, err_2_abs

    def save2csvtxt(self, path_dir_ana, file_pre, label, pred, diff, diff_abs, err, err_abs, err_2, err_2_abs):
        """
        save the prediction/label and there difference, error to csv and txt files
        input:
            path_dir_ana : the path of directory for output files
            file_pre : the prefix of output files, set by user
            label, pred, diff, diff_abs, err, err_abs, err_2, err_2_abs: data to output (numpy array)
        return:
            none
            the files titled file_pre + 'pred2.csv' / file_pre + 'pred2.txt' are saved to directory 'path_dir_ana'
        """
        os.makedirs(path_dir_ana, exist_ok=True)

        # 预测结果文件名和路径
        file_pred_txt = file_pre + '_pred2.txt'
        file_pred_csv = file_pre + '_pred2.csv'
        path_file_pred_txt = os.path.join(path_dir_ana, file_pred_txt)
        path_file_pred_csv = os.path.join(path_dir_ana, file_pred_csv)

        pred_ana2csv(label, pred, diff, diff_abs, err, err_abs, err_2, err_2_abs, path_file_pred_csv)
        self.pred_ana2txt(label, pred, diff, diff_abs, err, err_abs, err_2, err_2_abs, path_file_pred_txt)

    def pred_ana2txt(self, label, pred, diff, diff_abs, err, err_abs, err_2, err_2_abs, file):
        """
        将预测数据存为txt文件
        """
        global n_lab_self, n_lab_after_dot, n_lab_before_dot
        global n_pred_self, n_pred_after_dot, n_pred_before_dot
        global n_pred_abs_self, n_pred_abs_after_dot, n_pred_abs_before_dot
        global n_diff_self, n_diff_after_dot, n_diff_before_dot
        global n_diff_abs_self, n_diff_abs_after_dot, n_diff_abs_before_dot
        global n_err_self, n_err_after_dot
        global n_err_abs_self, n_err_abs_after_dot
        global n_err_2_self, n_err_2_after_dot
        global n_err_2_abs_self, n_err_2_abs_after_dot

        fout = open(file, 'w')

        # set the ourput of prediction value precision
        # the delta digit after dot between  label value and prediciton
        # for example ,if label is 1.2, delta =2, then the prediction will be precise to be 1.234
        # delta  = 3
        str_no = 'No.'
        str_lab = 'Label'
        str_pred = 'Predict'
        str_diff = 'Diff'
        str_diff_abs = '|Diff|'
        str_err = 'Error'
        str_err_abs = '|Error|'
        str_err_2 = 'Error_2'
        str_err_2_abs = '|Error_2|'

        # n_xx 是输出格式中各列的占位数
        n_maxnd_label = len(str(label.size))  # label的个数，如1万，则需要5位数
        n_no = max(n_maxnd_label, len(str_no))

        n_lab = max(n_lab_self, len(str_lab))
        n_pred = max(n_pred_self, len(str_pred))
        n_diff = max(n_diff_self, len(str_diff))
        n_diff_abs = max(n_diff_abs_self, len(str_diff_abs))
        n_err = max(n_err_self, len(str_err))
        n_err_abs = max(n_err_abs_self, len(str_err_abs))
        n_err_2 = max(n_err_2_self, len(str_err_2))
        n_err_2_abs = max(n_err_2_abs_self, len(str_err_2_abs))

        # print(f"{'*' * 5:>5} Label   : {dir_label}/{file_label}")
        print(f"{'*' * 5:>5} Label   : {dir_label}/{file_label}", file=fout)
        # print(f"Total   : {label.size} data")
        print(f"Total   : {label.size} data", file=fout)

        # print(
        #     f"\n{str_no:>{n_no}} {str_lab:>{n_lab}} {str_pred:>{n_pred}} {str_diff:>{n_diff}} {str_diff_abs:>{n_diff_abs}} {str_err:>{n_err}} {str_err_abs:>{n_err_abs}} {str_err_2:>{n_err_2}} {str_err_2_abs:>{n_err_2_abs}}")
        print(
            f"\n{str_no:>{n_no}} {str_lab:>{n_lab}} {str_pred:>{n_pred}} {str_diff:>{n_diff}} {str_diff_abs:>{n_diff_abs}} {str_err:>{n_err}} {str_err_abs:>{n_err_abs}} {str_err_2:>{n_err_2}} {str_err_2_abs:>{n_err_2_abs}}",
            file=fout)
        # print(
        #     f"{'-' * (n_no + 1 + n_lab + 1 + n_pred + 1 + n_diff + 1 + n_diff_abs + 1 + n_err + 1 + n_err_abs + 1 + n_err_2 + 1 + n_err_2_abs)}")
        print(
            f"{'-' * (n_no + 1 + n_lab + 1 + n_pred + 1 + n_diff + 1 + n_diff_abs + 1 + n_err + 1 + n_err_abs + 1 + n_err_2 + 1 + n_err_2_abs)}",
            file=fout)
        for i in range(pred.size):
            # print(
            #     f"{i:>{n_no}d} {label[i][0]:>{n_lab}.{n_lab_after_dot}f} {pred[i][0]:>{n_pred}.{n_pred_after_dot}f} {diff[i][0]:>{n_diff}.{n_diff_after_dot}f} {diff_abs[i][0]:>{n_diff_abs}.{n_diff_abs_after_dot}f} {err[i][0]:>{n_err}.{n_err_after_dot}E} {err_abs[i][0]:>{n_err_abs}.{n_err_after_dot}E} {err_2[i][0]:>{n_err_2}.{n_err_2_after_dot}E} {err_2_abs[i][0]:>{n_err_2_abs}.{n_err_2_after_dot}E}")
            print(
                f"{i:>{n_no}d} {label[i][0]:>{n_lab}.{n_lab_after_dot}f} {pred[i][0]:>{n_pred}.{n_pred_after_dot}f} {diff[i][0]:>{n_diff}.{n_diff_after_dot}f} {diff_abs[i][0]:>{n_diff_abs}.{n_diff_abs_after_dot}f} {err[i][0]:>{n_err}.{n_err_after_dot}E} {err_abs[i][0]:>{n_err_abs}.{n_err_after_dot}E} {err_2[i][0]:>{n_err_2}.{n_err_2_after_dot}E} {err_2_abs[i][0]:>{n_err_2_abs}.{n_err_2_after_dot}E}",
                file=fout)

        fout.close()

class ShowPredictLabelPlotThread(QThread):
    # 创建自定义信号
    get_data_signal = pyqtSignal(str)

    def __init__(self, signal):
        super().__init__()
        self.plot_path_signal = signal

    def run(self):
        # print("ShowPredictPlotThread线程在执行...")

        self.name = self.data.split('_')[1]
        # print(self.name) # chi

        set_parameters_analysis = [{'index': 1
                                    # 源文件夹
                                       , 'dir_pred': './../prediction'
                                    # 源文件名
                                       , 'file_pred': 'au150_DW_ann_' + self.name + '_cr1_200_20_1_pred.txt'
                                    # 对应label值文件夹
                                       , 'dir_label': './../datasets'
                                    # 对应label值文件名
                                       , 'file_label': 'Au_cr1_test.txt'
                                    # 目的文件夹
                                       , 'dir_ana': './../prediction_analysis'
                                    # 目的文件名前缀
                                       , 'file_pre': 'au150_DW_ann_' + self.name + '_cr1_200_20_1'

                                       , 'feature_index': self.name
                                       , 'label_index': 'CR1'
                                    # label的类型:
                                    # 'd' : 离散的，即值是离散分布的，不是连续的，比如配位数（都是整数）
                                    # 'c' : 连续的，即值是连续分布的，比如键长
                                    # 注意：由于这里采用的都是回归方法，因此预测值（prediction）类型都是连续的（c）
                                       , 'dict_label': {
                'type': 'c'
                , 'step': 0.02
            }

                                    # step:在对label值和预测值统计时，设置的间隔，即多大间隔的label值或预测值会统计在一起
                                    # 注意：如果类型是离散的，则‘step’的值程序不会使用，但用户仍然需要对其进行设置，不能删除！！
                                    # step_err : 用于预测误差分析统计时的间隔设置，一般是0.01（1%），如果预测误差很大，为了绘图方便，可以设置大点，如0.05（5%）等。
                                    # step_err_2 : 用于预测误差(error_2)分析统计时的间隔设置，一般是0.1（10%），如果预测误差很大，为了绘图方便，可以设置大点，如0.3（30%）等。
                                       , 'dict_pred': {
                'step': 0.02
                , 'step_err': 0.01
                , 'step_err_2': 0.1
            }
                                    }]

        for ith, param in enumerate(set_parameters_analysis):
            # 读取用户设置的参数
            self.read_param_analysis(param)

            # 标签路径
            path_file_label = os.path.join(os.curdir, dir_label, file_label)
            # print(path_file_label) # .\./../datasets\Au_cr1_test.txt
            # 源文件路径
            path_file_pred = os.path.join(os.curdir, dir_pred, file_pred)
            # print(path_file_pred) # .\./../prediction\au150_DW_ann_chi_cr1_200_20_1_run_2023_09_27-10_29_11_pred.txt

            # 检查预测文件、原值文件是否存在
            list_path = [path_file_label, path_file_pred]
            check_path_exist(list_path)

            # 导入feature和label的数据
            self.label, self.pred = read_label_pred(path_file_label, path_file_pred)

            self.path_dir_ana = os.path.join(os.curdir, dir_ana)

            # 直接比较label 和 prediction
            dict_plot_compareall = [{'ind': 0, 'W_plot_label': True, 'W_plot_pred': True, 'W_plot_diff': False}]
            for i, data in enumerate(dict_plot_compareall):
                ind = data['ind']
                W_plot_label = data['W_plot_label']
                W_plot_pred = data['W_plot_pred']
                W_plot_diff = data['W_plot_diff']
                self.prediction_compare_all(W_plot_label, W_plot_pred, W_plot_diff, ind)

        # 读取生成的整体分析
        file_name = 'au150_DW_ann_' + self.name + '_cr1_200_20_1_prediction_compare_CR1_0.png'
        # 源文件夹
        dir_data = './../prediction_analysis'
        png_path = os.path.join(dir_data, file_name)

        exist = os.path.exists(png_path)
        if not exist:
            print(f"** Error!! cannot find file {png_path}! **")

        # 传给主线程
        self.plot_path_signal.emit(png_path)

    def get_data(self, data):
        self.data = data

    def read_param_analysis(self, param):
        global dir_pred, file_pred, dir_label, file_label, dir_ana, file_pre
        global feature_index, label_index, dict_label, dict_pred

        dir_pred = param['dir_pred']
        file_pred = param['file_pred']
        dir_label = param['dir_label']
        file_label = param['file_label']
        dir_ana = param['dir_ana']
        file_pre = param['file_pre']
        feature_index = param['feature_index']
        label_index = param['label_index']
        dict_label = param['dict_label']
        dict_pred = param['dict_pred']

    def prediction_compare_all(self, W_plot_label, W_plot_pred, W_plot_diff, ind=0):
        """
        show the prediction verse real data in test datasets series
        """
        global file_pre, label_index
        global diff

        file = file_pre + f"_prediction_compare_{label_index}_{ind}"
        x = range(self.pred.size)
        # print(x) # range(0, 595)

        # plt.figure(figsize=(20, 16))
        plt.figure(figsize=(6, 5))
        if W_plot_label:
            plt.plot(x, self.label.flatten()[self.label.flatten().argsort()], 'o-', color='r', linewidth=0.1, markersize=2,
                     label=f"true_{label_index}")
        if W_plot_pred:
            plt.plot(x, self.pred.flatten()[self.label.flatten().argsort()], 'o-', linewidth=0.1, markersize=3,
                     label=f"pred_{label_index}")
        plt.xlabel('Test series', fontsize=12)
        if W_plot_label and W_plot_pred:
            plt.ylabel(f'{label_index}', fontsize=12)
        else:
            pass
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.title(f"{label_index} Regression Performance", fontsize=20)
        save_fig(self.path_dir_ana, file)
        plt.close()

# 主线程
class MyWindow(QMainWindow):
    # 创建自定义信号
    to_plainEdit_signal = pyqtSignal(str)
    train_end_signal = pyqtSignal(str)
    plot_path_signal = pyqtSignal(str)
    plot_model_path_signal = pyqtSignal(str)
    plot_loss_path_signal = pyqtSignal(str)
    plot_prediction_label_path_signal = pyqtSignal(str)
    predict_end_signal = pyqtSignal(str)
    modify_model_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.training_thread = None  # 训练线程
        self.init_ui()

    def init_ui(self):
        self.ui = uic.loadUi("./_internal/GUI.ui")
        # self.ui = uic.loadUi("GUI.ui")

        # 菜单栏控件
        self.action_quit = self.ui.action_quit
        self.action_about = self.ui.action_about

        # 使用QAction的triggered信号连接操作的信号与槽函数
        self.action_quit.triggered.connect(self.to_quit)
        self.action_about.triggered.connect(self.to_about)

        # 数据集及标签控件
        self.btn_get_data = self.ui.btn_get_data
        self.label_data_name = self.ui.label_data_name
        self.btn_get_label = self.ui.btn_get_label
        self.label_label_name = self.ui.label_label_name

        self.btn_data_analysis = self.ui.btn_data_analysis
        self.btn_label_analysis = self.ui.btn_label_analysis
        self.btn_label_plot = self.ui.btn_label_plot
        self.plainTextEdit = self.ui.plainTextEdit
        # self.plainTextEdit.setReadOnly(True)

        # 此处创建了一个图片窗口实例，解决出现重复窗口的问题
        self.image_dialog = ImageDialog()

        # 绑定信号与槽函数
        self.btn_get_data.clicked.connect(self.selectData)
        self.btn_get_label.clicked.connect(self.selectLabel)
        self.btn_data_analysis.clicked.connect(self.data_analysis)
        self.btn_label_analysis.clicked.connect(self.label_analysis)
        self.btn_label_plot.clicked.connect(self.label_plot)

        # 神经网络训练控件
        self.comboBox_model = self.ui.comboBox_model
        self.lineEdit_layer = self.ui.lineEdit_layer
        self.lineEdit_node = self.ui.lineEdit_node
        # self.lineEdit_split = self.ui.lineEdit_split
        self.lineEdit_epoch = self.ui.lineEdit_epoch

        self.btn_modify_model = self.ui.btn_modify_model
        self.btn_train = self.ui.btn_train
        self.btn_model = self.ui.btn_model
        self.btn_loss= self.ui.btn_loss
        self.btn_pre_label = self.ui.btn_pre_label
        self.btn_pre_analysis = self.ui.btn_pre_analysis

        self.btn_stop = self.ui.btn_stop
        self.btn_again = self.ui.btn_again

        # 绑定信号与槽函数
        self.btn_modify_model.clicked.connect(self.modify_model)
        self.btn_train.clicked.connect(self.nn_train)
        self.btn_model.clicked.connect(self.show_model)
        self.btn_loss.clicked.connect(self.show_loss)
        self.btn_pre_label.clicked.connect(self.show_pre_label)
        self.btn_pre_analysis.clicked.connect(self.show_pre_analysis)

        self.btn_stop.clicked.connect(self.train_stop)
        self.btn_again.clicked.connect(self.train_again)

        # 模型预测控件
        self.comboBox_predict_model = self.ui.comboBox_predict_model

        self.btn_get_test = self.ui.btn_get_test
        self.label_test_name = self.ui.label_test_name
        self.btn_predict = self.ui.btn_predict
        self.btn_prediction_label_plot = self.ui.btn_prediction_label_plot
        self.btn_prediction_analysis = self.ui.btn_prediction_analysis
        self.btn_again_predict = self.ui.btn_again_predict

        # 绑定信号与槽函数
        self.btn_get_test.clicked.connect(self.selectTest)
        self.btn_predict.clicked.connect(self.predict)
        self.btn_prediction_label_plot.clicked.connect(self.show_prediction_label_plot)
        self.btn_prediction_analysis.clicked.connect(self.show_prediction_analysis)
        self.btn_again_predict.clicked.connect(self.predict_again)

    # 模型预测模块
    def predict_again(self):
        self.btn_predict.setEnabled(True)
        self.btn_prediction_label_plot.setEnabled(False)
        self.btn_prediction_analysis.setEnabled(False)
        self.btn_again_predict.setEnabled(False)

    def show_prediction_analysis(self):
        # 获取测试集
        data = self.label_test_name.text()
        # print(data) # Au_chi_test

        if data:
            self.my_thread = ShowPredictAnalysisThread(self.to_plainEdit_signal)
            self.my_thread.get_data_signal.connect(self.my_thread.get_data)
            self.my_thread.get_data_signal.emit(data)  # 此处发射信号相当于调用了其对应的get_data函数
            self.my_thread.start()

            self.to_plainEdit_signal.connect(self.show_data)

    def plot_prediction_label(self,png_path):
        # print(png_path)

        self.image_dialog.set_image(png_path)
        self.image_dialog.exec_()

    def show_prediction_label_plot(self):
        # 获取测试集
        data = self.label_test_name.text()
        # print(data) # Au_chi_test

        if data:
            self.my_thread = ShowPredictLabelPlotThread(self.plot_prediction_label_path_signal)
            self.my_thread.get_data_signal.connect(self.my_thread.get_data)
            self.my_thread.get_data_signal.emit(data)
            self.my_thread.start()

            self.plot_prediction_label_path_signal.connect(self.plot_prediction_label)

    def show_predict_end(self,data):
        data = data
        self.plainTextEdit.setPlainText(data)
        # 设置控件显示
        self.btn_prediction_label_plot.setEnabled(True)
        self.btn_prediction_analysis.setEnabled(True)
        self.btn_again_predict.setEnabled(True)

    def predict(self):
        # 获取测试集
        data1 = self.label_test_name.text()
        # print(data) # Au_chi_test

        model = self.comboBox_predict_model.currentText()
        # print(model)

        if data1:
            test_name = data1.split('_')[1]
            # print(test_name)
            model_name = model.split('_')[1]
            # print(model_name)
            data = self.test_get
            # print(data) # E:/Code/XASDAML/UI/datasets/Au_chi_train.txt
            dict = {"path": data, "model": model, "data": data1}
            test_model_json = json.dumps(dict)

            if model == "current_training_model":
                # 判断当前模型是否存在（判断./../nn_train/au150_DW_ann_chi_cr1_200_20_1.h5是否存在）
                file_path = './../nn_train/au150_DW_ann_chi_cr1_200_20_1.h5'
                exist = os.path.exists(file_path)
                # print(exist)
                if exist and self.label_data_name.text() and self.label_label_name.text():
                    if test_name == self.label_data_name.text().split('_')[1]:
                        # 将训练按钮变灰
                        self.btn_predict.setEnabled(False)
                        self.plainTextEdit.setPlainText("正在预测，请等待...")

                        self.my_thread = PredictThread(self.predict_end_signal)
                        self.my_thread.get_data_signal.connect(self.my_thread.get_data)

                        self.my_thread.get_data_signal.emit(test_model_json)  # 此处发射信号相当于调用了其对应的get_data函数
                        self.my_thread.start()

                        self.predict_end_signal.connect(self.show_predict_end)
                    else:
                        QMessageBox.warning(self, 'Error', '模型与测试集不匹配！')
                else:
                    QMessageBox.warning(self, 'Error', '没有当前模型！')

            else:
                if test_name == model_name:
                    # 将训练按钮变灰
                    self.btn_predict.setEnabled(False)
                    self.plainTextEdit.setPlainText("正在预测，请等待...")

                    self.my_thread = PredictThread(self.predict_end_signal)
                    self.my_thread.get_data_signal.connect(self.my_thread.get_data)

                    self.my_thread.get_data_signal.emit(test_model_json)  # 此处发射信号相当于调用了其对应的get_data函数
                    self.my_thread.start()

                    self.predict_end_signal.connect(self.show_predict_end)
                else:
                    QMessageBox.warning(self, 'Error', '模型与测试集不匹配！')

        else:
            QMessageBox.warning(self, 'Error', '测试集不存在！')

    def selectTest(self):
        self.test_get, _ = QFileDialog.getOpenFileName(self, 'Select File')
        if self.test_get:
            file_name = self.test_get.split("/")[-1]
            label_test_name = file_name[:-4]
            self.label_test_name.setText(label_test_name)

    # 神经网络训练模块
    def train_again(self):
        self.btn_train.setEnabled(True)
        self.btn_model.setEnabled(False)
        self.btn_loss.setEnabled(False)
        self.btn_pre_label.setEnabled(False)
        self.btn_pre_analysis.setEnabled(False)
        self.btn_predict.setEnabled(False)
        self.btn_again.setEnabled(False)

        self.btn_prediction_label_plot.setEnabled(False)
        self.btn_prediction_analysis.setEnabled(False)
        self.btn_again_predict.setEnabled(False)

    def train_stop(self):
        print("训练暂停")

        self.btn_stop.setEnabled(False)

        self.my_thread = StopTrainThread(self.training_thread)
        self.my_thread.start()

        # 等待1秒（因为点完中断训练就点再次训练有时会退出）
        time.sleep(1)
        self.btn_again.setEnabled(True)

    def show_pre_analysis(self):
        # print("show_pre_analysis")
        data = self.label_data_name.text().split("_")[1]

        if data:
            self.my_thread = ShowPreAnalysisThread(self.to_plainEdit_signal)
            self.my_thread.get_combobox_signal.connect(self.my_thread.get_data)

            self.my_thread.get_combobox_signal.emit(data)
            self.my_thread.start()

            self.to_plainEdit_signal.connect(self.show_data)

    def show_pre_label(self):
        # print("show_pre_label")
        data = self.label_data_name.text().split("_")[1]

        if data:
            self.my_thread = ShowPreLabelThread(self.to_plainEdit_signal)
            self.my_thread.get_combobox_signal.connect(self.my_thread.get_data)

            self.my_thread.get_combobox_signal.emit(data)
            self.my_thread.start()

            self.to_plainEdit_signal.connect(self.show_data)

    def plot_loss(self,png_path):
        self.image_dialog.set_image(png_path)
        self.image_dialog.exec_()

    def show_loss(self):
        # print("show_loss")
        data = self.label_data_name.text().split("_")[1]

        if data:
            self.my_thread = ShowLossThread(self.plot_loss_path_signal)
            self.my_thread.get_combobox_signal.connect(self.my_thread.get_data)
            self.my_thread.get_combobox_signal.emit(data)
            self.my_thread.start()

            self.plot_loss_path_signal.connect(self.plot_loss)

    def plot_model(self,png_path):
        self.image_dialog.set_image(png_path)
        self.image_dialog.exec_()

    def show_model(self):
        # print("show_model")
        data = self.label_data_name.text().split("_")[1]

        if data:
            self.my_thread = ShowModelThread(self.plot_model_path_signal)
            self.my_thread.get_combobox_signal.connect(self.my_thread.get_data)
            self.my_thread.get_combobox_signal.emit(data)
            self.my_thread.start()

            self.plot_model_path_signal.connect(self.plot_model)

    def show_train_end(self,data):
        self.btn_stop.setEnabled(False)

        self.plainTextEdit.setPlainText(data)
        # 设置控件显示
        self.btn_model.setEnabled(True)
        self.btn_loss.setEnabled(True)
        self.btn_pre_label.setEnabled(True)
        self.btn_pre_analysis.setEnabled(True)
        self.btn_predict.setEnabled(True)
        self.btn_again.setEnabled(True)

    def outputWritten(self, text):
        cursor = self.plainTextEdit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.plainTextEdit.setTextCursor(cursor)
        self.plainTextEdit.ensureCursorVisible()

    def nn_train(self):
        # 获取数据集及标签
        data = self.label_data_name.text()

        # 标签没用上训练过程中
        label = self.label_label_name.text()

        # 获取参数和节点数
        layer = self.lineEdit_layer.text()
        node = self.lineEdit_node.text().split(';')
        # print(node) # ['200', '20', '1']
        lenth = len(node)

        epoch = self.lineEdit_epoch.text()

        if not data:
            QMessageBox.warning(self, 'Error', '数据集不存在！')

        if not label:
            QMessageBox.warning(self, 'Error', '标签不存在！')

        if int(layer) == lenth:
            if data and label:
                # 下面将输出重定向到plainTextEdit中
                sys.stdout = EmittingStr(textWritten=self.outputWritten)
                sys.stderr = EmittingStr(textWritten=self.outputWritten)

                self.plainTextEdit.clear()

                # 将训练按钮变灰
                self.btn_train.setEnabled(False)
                self.btn_stop.setEnabled(True)

                data_path = self.data_get
                label_path = self.label_get

                dict = {"data": data, "data_path":data_path, "label_path":label_path, "layer": layer, "node": node, "epoch": epoch}
                data_node_json = json.dumps(dict)

                print("正在训练，请等待...\n")
                # 开一个子线程进行神经网络训练
                # 需要传递data,没有传递label,因为label不可变
                self.training_thread = NNTrainThread(self.train_end_signal)
                self.training_thread.get_data_signal.connect(self.training_thread.get_data)
                self.training_thread.get_data_signal.emit(data_node_json)  # 此处发射信号相当于调用了其对应的get_data函数
                self.training_thread.start()

                self.train_end_signal.connect(self.show_train_end)
        else:
            QMessageBox.warning(self, 'Error', '层数与每层节点数不匹配！')

    def modify(self,model_info_json):
        model_info_json = json.loads(model_info_json)

        layer = model_info_json.get("layer")
        node_list = model_info_json.get("node_list")

        nodes = ";".join(str(num) for num in node_list)

        self.lineEdit_layer.setText(str(layer))
        self.lineEdit_node.setText(nodes)

    def modify_model(self):
        self.modify_model_signal.connect(self.modify)
        self.model_dialog = ModelDialog(self.modify_model_signal)
        font = QFont("Arial", 12)
        self.model_dialog.setFont(font)
        self.model_dialog.exec_()

    # 数据集及标签展示模块
    def plot_label(self,png_path):
        print(png_path)

        # 显示图片弹出一个框
        self.image_dialog.set_image(png_path)
        self.image_dialog.exec_()

    def label_plot(self):
        if self.label_label_name.text():
            label = self.label_get
            # print("label",label)
            self.my_thread = PlotLabelThread(self.plot_path_signal)
            self.my_thread.get_combobox_signal.connect(self.my_thread.get_data)

            self.my_thread.get_combobox_signal.emit(label)
            self.my_thread.start()

            self.plot_path_signal.connect(self.plot_label)
        else:
            QMessageBox.warning(self, 'Error', '标签不存在！')

    def label_analysis(self):
        if self.label_label_name.text():
            label = self.label_get
            # 需要将label传给子线程
            # 创建新线程，让它完成找到cn1.txt文件地址,并进行分析的操作，注意使用相对路径

            # 创建子线程时，将主线程创建的自定义信号传给了子线程
            self.my_thread = AnalysisThread(self.to_plainEdit_signal)
            self.my_thread.get_combobox_signal.connect(self.my_thread.get_data)

            self.my_thread.get_combobox_signal.emit(label)  # 此处发射信号相当于调用了其对应的get_data函数
            self.my_thread.start()  # 开始线程

            self.to_plainEdit_signal.connect(self.show_data)
        else:
            QMessageBox.warning(self, 'Error', '标签不存在！')

    def selectLabel(self):
        self.label_get, _ = QFileDialog.getOpenFileName(self, 'Select File')
        if self.label_get:
            file_name = self.label_get.split("/")[-1]
            label_label_name = file_name[:-4]
            self.label_label_name.setText(label_label_name)

    # 共用的
    def show_data(self,data):
        data = data
        self.plainTextEdit.setWordWrapMode(QTextOption.NoWrap)  # 设置不换行
        self.plainTextEdit.setPlainText(data)

    def data_analysis(self):
        if self.label_data_name.text():
            data = self.data_get
            # 需要将data传给子线程
            # 创建新线程，让它完成找到chi.txt文件地址,并进行分析的操作，注意使用相对路径

            # 创建子线程时，将主线程创建的自定义信号传给了子线程
            self.my_thread = AnalysisThread(self.to_plainEdit_signal)
            self.my_thread.get_combobox_signal.connect(self.my_thread.get_data)

            self.my_thread.get_combobox_signal.emit(data)  # 此处发射信号相当于调用了其对应的get_data函数
            self.my_thread.start()  # 开始线程

            self.to_plainEdit_signal.connect(self.show_data)
        else:
            QMessageBox.warning(self, 'Error', '数据集不存在！')

    def selectData(self):
        self.data_get, _ = QFileDialog.getOpenFileName(self, 'Select File')
        # print(self.data_get)
        if self.data_get:
            file_name = self.data_get.split("/")[-1]
            label_data_name = file_name[:-4]
            self.label_data_name.setText(label_data_name)

    def to_quit(self):
        # print("quit")
        QApplication.quit()

    def to_about(self):
        QMessageBox.about(self, 'About XASDAML', 'XASDAML v0.8,Copyright@Zhao Haifeng(zhaohf@ihep.ac.cn), '
                                                 'Platform of Advanced Photon Source Technology R&D(PAPS), '
                                                 'Institute of High Energy Physics, Chinese Academy of Science, Beijing 100049, China. ')


if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = MyWindow()
    w.ui.show()

    app.exec()
