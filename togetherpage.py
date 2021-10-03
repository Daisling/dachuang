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

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

import logging  # 日志记录

app = Flask(__name__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('sample.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

#######################################

basedir = os.path.abspath(r"F:\1\dachuang\merge_two_text_before\static\images\testpicture")  # 用户上传的图片都会先存进去
savedir = os.path.abspath(r"F:\1\dachuang\merge_two_text_before\static\images\savepicture")  # 预测结果大于80%的存进去
savedir2 = os.path.abspath(r"F:\1\dachuang\merge_two_text_before\static\images\train2")  # 自己标的数据

label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
              8: 'ship', 9: 'truck'}

app.config['SECRET_KEY'] = 'Thisissupposedtobesecret!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///F:\\1\\dachuang\\merge_two_text_before\\database.db'
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

traindir = os.path.abspath(r"F:\1\dachuang\merge_two_text_before\static\images\train2")  # 用来训练的数据
testdir = os.path.abspath(r"F:\1\dachuang\merge_two_text_before\static\images\test")

x_test = trainmodel.getx_test(testdir)  # x,y都是测试集来自F:\mini_cifar10\test
y_test = trainmodel.gety_test(testdir)

y_foracc = trainmodel.gety_foracc(y_test)

modelcifar10 = load_model('VGG16.h5')

global dataset
dataset = []
##############以下用于图表
count = 0  # 用户标注次数
global countlist
countlist = [0]

global losslist
losslist = []
global singlelist
singlelist = []
global doublelist
doublelist = []
global threelist
threelist = []


##############

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class LoginForm(FlaskForm):
    username = StringField('用户', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('密码', validators=[InputRequired(), Length(min=8, max=80)])


class RegisterForm(FlaskForm):
    email = StringField('邮箱', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('用户', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('密码', validators=[InputRequired(), Length(min=8, max=80)])



# 选择哪个模型部分
@app.route('/choosepage')
@login_required
def choosepage():
    return render_template("choosepage.html", name=current_user.username)


@app.route('/', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user)  # 有了这个之后才会变到choosepage 创建用户session
                return redirect(url_for('choosepage'))

        return '<h1>Invalid username or password</h1>'
        # return '<h1>' + form.username.data + ' ' + form.password.data + '</h1>'

    return render_template('login2.html', form=form)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return render_template('warn.html')

    return render_template('signup2.html', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/')


@app.route("/cifar10model", methods=["POST"])
@login_required
def begincifar10():
    return render_template("cifar10begin.html")


# cifar10模型部分
@app.route("/uploadcifar10picture", methods=['GET', 'POST'])
@login_required
def uploadcifar10():
    global files
    files = request.files.getlist('file')  # 这里面是可以随便写吗

    for file in files:
        filename = file.filename

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

    return render_template('test2.html', index_list=index_list, html=html)


@app.route('/labelcifar10/<data_3>', methods=['GET', 'POST'])
@login_required
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
                logger.info('user:{} labled {} as {} '.format(current_user.username, data_3, label))
                global count
                print(count)
                count = count + 1
                countlist.append(count)

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

    return render_template('test2.html', index_list=index_list, html=html, eval=eval)


@app.route('/data', methods=["GET", "POST"])
def data():
    model2 = load_model("VGG16.h5")

    eval = trainmodel.evaluate(model2, x_test, y_test, y_foracc)  # 返回的顺序是：损失函数有多大，单标签准确度，双标签准确度，三标签准确度

    losslist.append(eval[0])
    singlelist.append(float(eval[1]))
    doublelist.append(float(eval[2]))
    threelist.append(float(eval[3]))

    print(losslist)
    print(eval[1])

    return jsonify({'count': countlist, 'accuracy': losslist, 'single_lable': singlelist, 'double_lable': doublelist,
                    'three_lable': threelist})


@app.route("/changelabelcifar10", methods=["POST"])
@login_required
def changelabelcifar10():
    return render_template("cifar10predictensure.html", name="wrong", name1="whatever")


if __name__ == '__main__':
    app.run(debug=True, threaded=False)  # 解决AttributeError: '_thread._local' object has no attribute 'value'
