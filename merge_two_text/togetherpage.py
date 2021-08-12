import matplotlib.pyplot as plt
import cv2
import random
import os
from page_utils import Pagination

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify, flash
from flask_bootstrap import Bootstrap
import trainmodel  # 一些函数在这个文件里

basedir = os.path.abspath(r"F:\merge_two_text\static\images\testpicture")
savedir = os.path.abspath(r"F:\merge_two_text\static\images\savepicture")
savedir2 = os.path.abspath(r"F:\merge_two_text\static\images\train2")

label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
              8: 'ship', 9: 'truck'}

app = Flask(__name__)
Bootstrap(app)

app.config['SECRET_KEY'] = os.urandom(24)
# 设置为24位的字符,每次运行服务器都是不同的，所以服务器启动一次上次的session就清除。


traindir = os.path.abspath(r"F:\merge_two_text\static\images\train2")
testdir = os.path.abspath(r"F:\com\merge_two\static\images\test")

x_test = trainmodel.getx_test(testdir)  # x,y都是测试集来自F:\mini_cifar10\test
y_test = trainmodel.gety_test(testdir)

y_foracc = trainmodel.gety_foracc(y_test)

modelcifar10 = load_model('VGG16.h5')

global data
global dataset
dataset = []

upload_path = None  # 尝试了一下设全局，不行
fileName = None


# 选择哪个模型部分
@app.route("/")
def begin():
    return render_template("choosepage.html")


@app.route("/mnistmodel", methods=["POST"])
def beginmnist():
    return render_template("mnistbegin.html")


@app.route("/cifar10model", methods=["POST"])
def begincifar10():
    return render_template("cifar10begin.html")


# cifar10模型部分
@app.route("/uploadcifar10picture", methods=['GET', 'POST'])
def uploadcifar10():
    global files
    files = request.files.getlist('file')  # 这里面是可以随便写吗
    global filelist
    global urllist
    filelist = []
    urllist = []
    for file in files:
        filename = file.filename
        filetype = filename.split('.')[-1]

        print(filename)
        print(filetype)

        upload_path = os.path.join(basedir, secure_filename(filename))
        file.save(upload_path)  # 存testpicture

        img = load_img(upload_path, grayscale=False)
        img = np.array(img).reshape((1, 32, 32, 3))
        img = img.astype('float32') / 255

        predict = modelcifar10.predict(img)
        probablity = np.max(predict)

        predict = np.argmax(predict)
        result = format(label_dict[predict])

        if probablity >= 0.80:
            img_to_save = cv2.imread(upload_path)
            resultstr = str(result)
            save_path = os.path.join(savedir, resultstr)

            if os.path.isdir(save_path):

                cv2.imwrite(os.path.join(save_path, secure_filename(filename)), img_to_save)

            else:
                os.mkdir(save_path)
                cv2.imwrite(os.path.join(save_path, secure_filename(filename)), img_to_save)
        else:

            url = url_for("static", filename="images/testpicture/" + filename)

            print(url)

            print(probablity)

            data = [url, result, probablity, filename]

            print(data)

            dataset.append(data)

    return render_template('cifar10begin.html', msg='图片上传成功')


@app.route("/begintoexersize", methods=['GET', 'POST'])  # post隐式提交，get显示提交
def predict():
    pager_obj = Pagination(request.args.get("page", 1), len(dataset), request.path, request.args, per_page_count=8)
    print(request.path)
    print(request.args)

    index_list = dataset[pager_obj.start:pager_obj.end]
    html = pager_obj.page_html()

    eval = trainmodel.evaluate(modelcifar10, x_test, y_test, y_foracc)

    return render_template('test.html', index_list=index_list, html=html, eval=eval)


@app.route('/labelcifar10/<data_3>', methods=['GET', 'POST'])
def labelcifar10(data_3):



    label = request.form.get("label")

    if label:

        print(data_3)
        print(label)

        upload_path = os.path.join(basedir, secure_filename(data_3))

        img_to_save = cv2.imread(upload_path)

        save_path = os.path.join(savedir2, label)

        photoname = data_3

        if os.path.isdir(save_path):
            cv2.imwrite(os.path.join(save_path, photoname), img_to_save)
        else:
            os.mkdir(save_path)
            cv2.imwrite(os.path.join(save_path, photoname), img_to_save)

        for data in dataset:
            if data[3] == data_3:
                dataset.remove(data)

        x_train = trainmodel.newx_train(traindir)
        y_train = trainmodel.newy_train(traindir)

        trainmodel.train(x_train, y_train, x_test, y_test)
    model2 = load_model("VGG16.h5")



    eval = trainmodel.evaluate(model2, x_test, y_test, y_foracc)  # 返回的顺序是：损失函数有多大，单标签准确度，双标签准确度，三标签准确度

    pager_obj = Pagination(request.args.get("page", 1), len(dataset), request.path, request.args, per_page_count=8)
    print(request.path)
    print(request.args)

    index_list = dataset[pager_obj.start:pager_obj.end]
    html = pager_obj.page_html()

    return render_template('test.html', index_list=index_list, html=html, eval=eval)


# label是人工标注出来的标签，name1是图片路径


@app.route("/changelabelcifar10", methods=["POST"])
def changelabelcifar10():
    return render_template("cifar10predictensure.html", name="wrong", name1="whatever")


if __name__ == '__main__':
    app.run(debug=True, threaded=False)  # 解决AttributeError: '_thread._local' object has no attribute 'value'
