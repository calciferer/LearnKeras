import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.callbacks import LambdaCallback
from keras.initializers import Constant
from keras.layers import Dense
from keras.optimizers import SGD


class LinearRegression:
    """
    线性回归
    """

    x_train = None  # type:np.ndarray
    y_train = None  # type:np.ndarray

    def gen_data(self):
        """
        生成x，y数据。这里拟合一个线性函数 y = 2 * x + 1
        """

        # 生成训练集数据
        # x从0到10，生成101个
        x_train = np.linspace(0, 10, 101)
        y_train = x_train * 2 + 1
        self.x_train = x_train
        self.y_train = y_train
        return x_train, y_train

    def trainByKeras(self):
        """
        通过Keras来训练
        """
        model = Sequential()
        # 添加一个全连接层，units=1表示该层有一个神经元，input_dim=1表示x_train中每个样本的维度为1,初始值W=5,b=2
        model.add(Dense(units=1, input_dim=1, kernel_initializer=Constant(5), bias_initializer=Constant(2)))
        # 定义优化算法为sgd:随机梯度下降,定义损失函数为mse:均方误差
        model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')
        callback = LambdaCallback(
            on_batch_begin=lambda batch, logs: print(f"batch:{batch},para:{model.layers[0].get_weights()}"))
        model.fit(self.x_train, self.y_train, epochs=1, batch_size=1, callbacks=[callback], shuffle=False)
        # 训练完成后，获取模型的参数
        W, b = model.layers[0].get_weights()
        print(f"W:{W},b:{b}")
        # 获取模型的预测值
        y_pre = model.predict(lr.x_train)

        # 绘制train散点图
        plt.plot(lr.x_train, lr.y_train, 'b.', markersize=1)
        plt.plot(lr.x_train, y_pre, 'r-')
        plt.show()

    def trainByPurePython(self):
        """
        手动写具体实现
        """
        # 学习率
        alpha = 0.01
        # 轮次
        epoch = 1
        # 初始化W,b
        W = 5.0
        b = 2.0
        for step in range(epoch):
            print(f"开始第{step + 1}个epoch")
            # 在这个例子中，样本大小为101:
            # 如果BatchSize设置为1，那么就属于`SGD`随机梯度下降，取得每个样本都进行一次梯度下降，更新一次W, b
            # 如果BatchSize设置为101，那么就属于`BGD`批量梯度下降，每个epoch只进行一次梯度下降
            # 如果BatchSize设置为1至101中的某个数，比如30，那么就属于`MBGD(Mini-Batch)小批量梯度下降`，每经过指定的batchSize，进行一次梯度下降,每遍历batchSize个样本，称为一次`iteration（迭代）`。
            batch_size = 1
            W_grad_sum = 0.0  # 每个样本点对W偏导数的和
            b_grad_sum = 0.0  # 每个样本点对b偏导数的和
            total_loss = 0.0  # 总loss
            for index, (x, y) in enumerate(zip(self.x_train, self.y_train)):
                total_loss += (W * x + b - y) ** 2
                W_grad_sum += 2 * x * (W * x + b - y)
                b_grad_sum += 2 * (W * x + b - y)
                if (index + 1) % batch_size == 0 or index == len(self.x_train) - 1:
                    print(f"经过第{int(index / batch_size) + 1}次迭代,当前index={index}")
                    W -= alpha * (W_grad_sum / batch_size)  # 梯度和除以样本数
                    b -= alpha * (b_grad_sum / batch_size)  # 梯度和除以样本数
                    print(f'loss:{total_loss / batch_size},W={W},b={b}')
                    W_grad_sum = 0.0
                    b_grad_sum = 0.0
                    total_loss = 0.0

        # 训练完成
        print(f"W:{W},b:{b}")
        # 获取模型的预测值
        y_pre = W * self.x_train + b

        # 绘制train散点图
        plt.plot(lr.x_train, lr.y_train, 'b.', markersize=1)
        plt.plot(lr.x_train, y_pre, 'r-')
        plt.show()


if __name__ == '__main__':
    lr = LinearRegression()
    lr.gen_data()
    lr.trainByKeras()
