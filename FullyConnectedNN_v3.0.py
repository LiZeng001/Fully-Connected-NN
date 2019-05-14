"""
 ● Hi there, this is LiZeng's Python Work.
 ● 全连接神经网络 + v3.0
    ▸ 1. 实现全连接神经网络的BP算法
    ▸ 2. 网络参数：层数、神经元数、隐藏层激活函数 可配置
    ▸ 3. 加入Mini-batch GD
 ● from 2019-05-08
 ● Any comments or suggests will be appreciated!
 ● 如果你忍不住地想跟我交朋友: mofistova@qq.com

 ✦ 可扩展问题：
    ▸ 1. 目前写为函数形式，可重构为类
    ▸ 2. 说明：代码中公式对应知乎专栏：
         文1：https://zhuanlan.zhihu.com/p/64202456
         文2：https://zhuanlan.zhihu.com/p/61531989
         文3：https://zhuanlan.zhihu.com/p/65679447
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime

def sigmoid(x):
    """
    Sigmoid激活函数
    :param x: 输入数据，可以是标量、向量、矩阵. 自动broadcast
    :return: 激活后的值
    """
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivate(x):
    """
    Sigmoid函数求导
    :param x: 输入数据，可以是标量、向量、矩阵. 自动broadcast
    :return: σ'(x) = σ(x)[1 - σ(x)]
    """
    return sigmoid(x) * (1 - sigmoid(x))  # *为Hadamard积

def relu(x):
    """
    Relu激活函数
    :param x: 输入数据，可以是标量、向量、矩阵
    :return: 激活后的值
    """
    return np.maximum(x, 0)

def relu_derivate(x):
    """
    Relu函数求导
    :param x: 输入数据，可以是标量、向量、矩阵
    :return: 1 if x >= 0 else 0
    """
    try:
        float(x)  # 如果是标量则运行try部分
        res = 1 if x >= 0 else 0
        return res
    except TypeError:  # 如果是向量或矩阵则执行except部分
        x[x >= 0] = 1
        x[x < 0] = 0
        return x

def tanh(x):
    """
    tanh(双曲正切)激活函数
    :param x: 输入数据，可以是标量、向量、矩阵
    :return: 激活后的值
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_derivate(x):
    """
    tanh函数求导
    :param x: 输入数据，可以是标量、向量、矩阵
    :return: 1-tan^2(x)
    """
    return 1 - np.power(tanh(x), 2)

def activation(x, activation_choice):
    """可设置的激活函数. 目前加入最经典的3中类型"""
    if activation_choice.lower() == 'sigmoid':
        return sigmoid(x)
    elif activation_choice.lower() == 'relu':
        return relu(x)
    elif activation_choice.lower() == 'tanh':
        return tanh(x)

def activation_derivation(x, activation_choice):
    """可设置的激活函数导数. """
    if activation_choice.lower() == 'sigmoid':
        return sigmoid_derivate(x)
    elif activation_choice.lower() == 'relu':
        return relu_derivate(x)
    elif activation_choice.lower() == 'tanh':
        return tanh_derivate(x)

def softmax(Y):
    """
    计算输出层的softmax
    :param Y: 输出层 z^L = WX + b，不做activation. 可以是向量、矩阵
    :return: softmax_score
    """
    exp_score = np.exp(Y)
    softmax_prbs = exp_score / np.sum(exp_score, axis=0, keepdims=True)  # 按行相加
    return softmax_prbs

def CrossEntropyLoss(softmax_prob, y):
    """
    计算Cross Entropy误差. 暂时不考虑正则项：根据L2-norm regularization的定义，对正则项的梯度其实就是矩阵元素求和
    :param softmax_prob:
    :param y: 真实标签
    :return: 交叉熵误差
    """
    m = softmax_prob.shape[1]  # 该次训练的数据量
    crosEnt = - np.sum(y * np.log(softmax_prob)) / m  # 注意是np.sum()，不是python自带的sum()
    return crosEnt

def params_init(layer_structure):
    """
    初始化网络参数
    :param layer_structure: 网络各层神经元个数，list. 第1层数量为单个数据特征维度
    :return: 从第2层到第L层的权重和偏置，Weights → list, biases → list. idx: [0,...,L-2], Weights[0] = W^2
    """
    np.random.seed(0)
    Weights = [np.random.randn(x, y) / np.sqrt(y) for x, y in zip(layer_structure[1:], layer_structure[: -1])]
    biases = [np.random.randn(x, 1) for x in layer_structure[1:]]
    return Weights, biases

def Feedforword(X, Weights, biases, activation_choice):
    """
    1次前向传播
    :param X: 输入数据，每个数据为1列，维度n1=layer_structure[0]; m个数据则为 n1 x m
    :param activation_choose: 隐藏层(第2,...,L-1层)神经元激活函数类型。输出层用softmax
    :return: zs → 各层神经元输入向量，列表；ys → 各层神经元激活值，列表，第1层zs[0]=ys[0]=X
    :return: Weights, biases：当前网络参数，供当前BackPropagation()更新参数
    """
    # 输入层神经元输入与激活值都认为是X，也存入列表，BP时需用到
    zs = [X]
    ys = [X]

    # 文2-式(2.2) 第2,...,L-1层. 输出层不做activation，做Softmax，与文2中略有改动. 参考文3-Chap2
    y = X
    for W, b in zip(Weights[: -1], biases[: -1]):  # 注意 z, y 均为n_l x K 维矩阵. K是每次Feed的数据量
        z = np.dot(W, y) + b
        zs.append(z)  # 存放隐藏层神经元输入
        y = activation(z, activation_choice)
        ys.append(y)  # 存放隐藏层神经元激活值(由activation_func激活)

    # 第L层 文2-式(3.9) 这里变量名比较啰嗦，主要是为了便于理解.
    z_L = np.dot(Weights[-1], y) + biases[-1]
    zs.append(z_L)  # 存放输出层神经元输入
    y_L = softmax(z_L)
    ys.append(y_L)  # 存放输出层神经元输出，由softmax激活

    return (zs, ys, Weights, biases)

def BackPropagation(zs, ys, y, Weights, biases, learning_rate, activation_choice):
    """
    当前Feedforward的反向传播，更新网络参数 文2-式(2.26)
    :param zs: 前向传播得到的各层输入
    :param ys: 前向传播得到的各层激活值
    :param y: 真实标签-one hot
    :param Weights: 当前需更新的权重
    :param biases: 当前需更新的偏置
    :return: 更新后的Weights，biases
    """
    # # 查看更新前参数 Weights
    # for W_idx, Weight in enumerate(Weights):
    #     print(W_idx, ": ", Weight)

    '''Step1 计算各层梯度并存储'''
    L = len(zs)
    dWs = []
    dbs = []
    m = ys[-1].shape[1]

    '''Step1.1 计算第L层的梯度，注意由于数据是一次输入多个，因此需要做梯度平均'''
    delta = ys[-1] - y  # 文3-式(2.1) δ^L 注意由于有m个数据，因此形成了矩阵Δ∈R^{n_L x m}.
    dW_L = np.dot(delta, ys[-2].T) / m  # 文3-式(2.1)
    dWs.append(dW_L)
    db_L = np.sum(delta, axis=1, keepdims=True) / m  # 文3-式(2.2) dE_db^L
    dbs.append(db_L)

    '''Step 1.2 计算公式中的W^l和b^l的(平均)梯度, l∈[L-1,L-2,...,2]'''
    for l in range(L-3, -1, -1):
        '''
        这里的idx会比较绕：
        公式中W和b的idx是2,3,...,L. 而在代码的Weights和biases的list中，对应idx是0,1,...L-2. 偏移量2
        公式中的层数idx是1,2,...,L. 而在代码中的idx是0,1,...,L-1，偏移量1. 也对应ys和zs的list的idx
        在递推代码部分，需要规定一个量的idx作为标准，其余量在此基础上调整. 这里以W和b的idx为基准
        因此，已经计算了W^{L}，实际idx为L-2，递推从公式第L-1层开始，对应代码中idx则为L-2-1,...,0
        此基准下，代码中W的idx=l —— 则公式中为l+2，因此对应层数和zs、ys的idx公式中l+2, 则代码中l+2-1 = l+1
        '''
        delta = np.dot(Weights[l + 1].T, delta) * activation_derivation(zs[l + 1], activation_choice)  # 文2-式(2.15) δ^l, l = L-1,...,2
        dW = np.dot(delta, ys[l].T) / m  # 计算dW平均梯度 文2-式(2.18)
        db = np.sum(delta, axis=1, keepdims=True) / m  # 计算db平均梯度 文2-式(2.19)
        dWs.append(dW)
        dbs.append(db)

    dWs.reverse()  # 由于是按从后往前递推计算顺序存放，反序排列，与Weights, biases顺序一致
    dbs.reverse()

    # # 查看平均梯度dWs
    # for dW_idx, dW in enumerate(dWs):
    #     print(dW_idx, ": ", dW)

    '''Step 2 更新参数'''
    Weights_updated = [W - learning_rate * dW for W, dW in zip(Weights, dWs)]
    biases_updated = [b - learning_rate * db for b, db in zip(biases, dbs)]

    # # 查看更新后参数 Ws
    # for W_idx, Weight in enumerate(Weights_updated):
    #     print(W_idx, ": ", Weight)

    return Weights_updated, biases_updated

def train_model(iterations, X, y, layer_structure, learning_rate, activation_choice):
    """
    通过上面Feedforward()和Backpropagation()训练网络-Batch GD
    :param iterations: 迭代次数
    :param layer_structure: 网络结构参数
    :param X: 训练数据
    :param y: 真实标签
    :param activation_choice: 隐藏层激活函数类型
    :return: 最终训练好的参数及误差
    """
    Weights, biases = params_init(layer_structure)
    costs = []
    """ 反复Feedforward/BackPropagation iterations 次来更新网络参数 """
    for i in range(iterations):
        zs, ys, Weights, biases = Feedforword(X, Weights, biases, activation_choice)
        Weights, biases = BackPropagation(zs, ys, y, Weights, biases, learning_rate, activation_choice)
        """ 计算误差 """
        cost = CrossEntropyLoss(ys[-1], y)
        costs.append(cost)
        if i % 1000 == 0:
            print("Loss at {}-th iteration is {}".format(i, cost))
    return Weights, biases, costs

def train_model_minibatch(epochs, mini_batch_size, X, y, layer_structure, learning_rate, activation_choice):
    """
    Mini-Batch GD 训练网络
    :param epochs: 全部数据batch的训练次数
    :param mini_batch_size: 如命名含义
    :param X: 输入数据，整个train_data
    :param y: 真实标签
    :param layerstructure: 网络结构参数
    :param activation_choice: 激活函数
    :return: 最终训练好的参数及误差
    """
    """ 初始网络参数 """
    Weights, biases = params_init(layer_structure)

    """ mini-batch 训练 """
    print("Each Epoch has {} Mini-Batches.".format(int(X.shape[1] / mini_batch_size)))
    costs = []
    global cost
    for i in range(epochs):
        """ 数据及标签置乱: 每个epoch都置乱一次 """
        permutation = np.random.permutation(y.shape[1])  # 置乱的index
        data_shuffled = X.T[permutation].T  # 输入数据为列向量，因此先变回行向量形式置乱，再转回列向量
        label_shuffled = y.T[permutation].T  # 同理，标签也为每列对应一个数据，应先转置，置乱后再转回来
        m = data_shuffled.shape[1]
        X_mini_batches = [data_shuffled[:, j: j + mini_batch_size] for j in range(0, m, mini_batch_size)]
        y_mini_batches = [label_shuffled[:, j: j + mini_batch_size] for j in range(0, m, mini_batch_size)]

        """  """
        for k, _ in enumerate(X_mini_batches):

            zs, ys, Weights, biases = Feedforword(X_mini_batches[k], Weights, biases, activation_choice)
            Weights, biases = BackPropagation(zs, ys, y_mini_batches[k], Weights, biases, learning_rate, activation_choice)

            """ 计算误差 注意Loss计算时数据量的选择 """
            cost = CrossEntropyLoss(ys[-1], y_mini_batches[k])  # 用mini-batch计算Loss
            # 用整个batch计算Loss
            # _, ys_batch, _, _ = Feedforword(X, Weights, biases, activation_choice)
            # cost = CrossEntropyLoss(ys_batch[-1], y)

            """ 不同分辨率记录Loss """
            # costs.append(cost)  # Resolution1: 每个Epoch，所有mini-batch记录一次

            if (i % 100 == 0) and (k % 1 == 0):  # Resolution2: 每100个epoch，所有mini记录一次. 可视化效果较好
            # if (i % 100 == 0) and (k % 3 == 0):  # Resolution3: 每100个epoch，每3个mini记录一次
                print("Loss at {}-th epoch's {}-th mini-batch is {}".format(i + 1, k + 1, cost))
                costs.append(cost)

        # costs_epoch.append(cost)  # Resolution4: 每个epoch第1个mini-batch记录一次

    return Weights, biases, costs

def predict(X, Weights, biases, activation_choice):
    """
    用训练好的模型预测结果
    :param X: test_data. 这里我们直接用了train_data
    :param Weights: 训练好的权重矩阵列表
    :param biases: 训练好的偏置向量列表
    :param activation_choice: 隐藏层激活函数
    :return: 预测结果
    """
    _, output, _, _ = Feedforword(X.T, Weights, biases, activation_choice)
    output = output[-1].T
    result = output / np.sum(output, axis=1, keepdims=True)
    return np.argmax(result, axis=1)

def plot_decision_boundary(pred_func):
    """ 绘制决策边界 """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='black')


if __name__=='__main__':
    """ 生成数据. 采用CS231 demo中的3分类数据 """
    np.random.seed(1)
    N = 200  # 每个类中的样本点，3类共N * K点，每点D=2维
    D = 2  # 每个点2个维度/属性
    K = 3  # 类别数
    X = np.zeros((N * K, D))  # 样本维度 (300, 2), data matrix (each row = single example)
    y = np.zeros(N * K)  # 类别标签
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j  # y即为标签
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)  # 数据可视化
    # plt.show()

    """ 对y进行one-hot编码 """
    expected_output = []
    for c in y:  # 类别数会将输出层单元数确定下来，这里是3维，则n_L = 3
        if c == 0:
            expected_output.append([1, 0, 0])
        elif c == 1:
            expected_output.append([0, 1, 0])
        else:
            expected_output.append([0, 0, 1])
    expected_output = np.array(expected_output).T  # 训练时使用的标签

    """ 设置模型结构参数 """
    layer_structure = [2, 25, 25, 3]
    activation_choice = 'tanh'
    iterations = 25001
    learning_rate = 0.008
    """ mini-batch参数 """
    epochs = 10001
    mini_batch_size = 20

    """ 训练网络，计时 """
    start_time = datetime.datetime.now()

    # Batch GD
    Weights_trained, biases_trained, costs = train_model(iterations, X.T, expected_output,
                                                         layer_structure, learning_rate, activation_choice)

    # # Mini-Batch GD
    # Weights_trained, biases_trained, costs = train_model_minibatch(epochs, mini_batch_size, X.T, expected_output,
    #                                                                layer_structure, learning_rate, activation_choice)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    time_cost = float(duration.seconds) + float(duration.microseconds * 1e-6)

    """计算模型精确度"""
    class_pred = predict(X, Weights_trained, biases_trained, activation_choice)
    accuracy = np.mean(class_pred == y)
    print("training accuracy: %.5f" % accuracy)

    """ 绘制决策边界和误差曲线 """
    plt.figure()
    plt.subplot(121)
    plot_decision_boundary(lambda x: predict(x, Weights_trained, biases_trained, activation_choice))

    plt.subplot(122)
    plt.scatter(list(range(len(costs))), costs, s=1, c='orange', marker='.')
    plt.xlabel('Iterations. Model Accuracy is %.5f' % accuracy)
    plt.ylabel('Cross Entropy Loss')
    plt.title('{}: Minimum Error is %.3f, Time Cost is %.3fs'.format(activation_choice.title()) % (min(costs), time_cost))
    plt.grid()
    plt.show()



