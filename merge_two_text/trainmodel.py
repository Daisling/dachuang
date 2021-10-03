from tensorflow.keras import optimizers
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout, Flatten, Dense
import os
import numpy as np
from PIL import Image
import heapq
from keras.models import load_model

def newx_train(traindir):
    trainimage = os.listdir(traindir)
    x_train = []
    for i in trainimage:
        readpath = os.path.join(traindir, i)
        label = os.path.split(readpath)[1]
        for j in os.listdir(readpath):
            picpath = os.path.join(readpath, j)
            img = Image.open(picpath).convert("RGB")
            x_train.append(np.array(img))

    return x_train

def newy_train(traindir):
    trainimage = os.listdir(traindir)
    y_train = []
    label_dict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7,
                  'ship': 8, 'truck': 9}
    for i in trainimage:
        readpath = os.path.join(traindir, i)
        label = os.path.split(readpath)[1]
        for j in os.listdir(readpath):
            labelnumber = label_dict.get(label)
            y_train.append(int(labelnumber))

    y_train = np.array(y_train)  # 将标签转化为numpy类型的数组
    return y_train

def getx_test(testdir):
    testimage = os.listdir(testdir)
    x_test=[]
    for i in testimage:
        readpath = os.path.join(testdir, i)
        label = os.path.split(readpath)[1]
        for j in os.listdir(readpath):
            picpath = os.path.join(readpath, j)
            img = Image.open(picpath).convert("RGB")
            x_test.append(np.array(img))

    return x_test

def gety_test(testdir):
    testimage = os.listdir(testdir)
    y_test = []
    label_dict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7,
                  'ship': 8, 'truck': 9}
    for i in testimage:
        readpath = os.path.join(testdir, i)
        label = os.path.split(readpath)[1]
        for j in os.listdir(readpath):
            labelnumber = label_dict.get(label)
            y_test.append(int(labelnumber))

    y_test = np.array(y_test)  # 将标签转化为numpy类型的数组
    return y_test

def gety_foracc(y_test):
    y_foracc=y_test
    return y_foracc

def train(x_train,y_train,x_test,y_test):
      x_train = np.array(x_train, dtype=np.float32) / 255
      x_test = np.array(x_test, dtype=np.float32) / 255
      y_train = to_categorical(y_train, 10)  # 独热编码
      y_test = to_categorical(y_test, 10)

      # 定义模型
      base_model = applications.VGG16(
          weights="imagenet", include_top=False,
          input_shape=(32, 32, 3))  # 预训练的VGG16网络，替换掉顶部网络

      for layer in base_model.layers[:15]:
          layer.trainable = False  # 冻结预训练网络前15层，最后的卷积神经网络块可训练

      top_model = Sequential()  # 自定义顶层网络
      top_model.add(Flatten(input_shape=base_model.output_shape[1:]))  # 将预训练网络展平
      top_model.add(Dense(32, activation='relu'))  # 全连接层，输入像素32
      top_model.add(Dropout(0.5))  # Dropout概率0.5
      top_model.add(Dense(10, activation='softmax'))  # 输出层，十分类

      model = Model(
          inputs=base_model.input,
          outputs=top_model(base_model.output))  # 新网络=预训练网络+自定义网络



      model.compile(
          loss='categorical_crossentropy',
          optimizer=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
          metrics=['accuracy'])  # 损失函数为分类交叉熵，优化器为SGD

      print(model.summary())

      # 训练&保存
      checkpointer = ModelCheckpoint(
          filepath='VGG16.h5', verbose=1, save_best_only=True)  # 保存最优模型

      model.fit(
          x=x_train,
          y=y_train,
          batch_size=1,
          epochs=1,
          verbose=1,
          callbacks=[checkpointer],
          validation_split=0.5,#validation_split用于在没有提供验证集的时候，按一定比例从训练集中取出一部分作为验证集
          shuffle=True)





def evaluate(model,x_test,y_test,y_foracc):
    label_dict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7,
                  'ship': 8, 'truck': 9}

    x_test = np.array(x_test, dtype=np.float32) / 255
    y_test = to_categorical(y_test, 10)

    predict = model.predict(x_test)
    total = len(predict)
    print(total)
    two = 0
    three = 0

    # 预测两个个标签的准确率
    n = 0
    for i in predict:
        top3 = heapq.nlargest(2, range(len(predict[n])), predict[n].take)
        for j in range(len(top3)):
            if (top3[j] == y_foracc[n]):
                two = two + 1
            else:
                two = two
        n = n + 1
    acc_two = two / total

    # 预测三个标签的准确率
    m = 0
    for i in predict:
        top3 = heapq.nlargest(3, range(len(predict[m])), predict[m].take)
        for j in range(len(top3)):
            if (top3[j] == y_foracc[m]):
                three = three + 1
            else:
                three = three
        m = m + 1
    acc_three = three / total

    # 测试集损失与准确率
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    return[test_loss,test_acc,acc_two,acc_three]  #返回的顺序是：损失函数，单标签准确度，双标签准确度，三标签准确度