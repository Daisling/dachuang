import json

from flask import request, jsonify, app, Flask, render_template

app = Flask(__name__)
@app.route("/")
def begin():
    return render_template("jstext.html")

@app.route('/aaa', methods=['POST'])
def aaa():
    data = json.loads(request.form.get('data'))
    a= data['a']
    b= data['b']
    print (a,b)
    # msg = bbb(a, b)#调用 bbb方法拿返回值
    msg =a,b
    return jsonify(msg)
if __name__ == '__main__':
    app.run(debug=True, threaded=False)  # 解决AttributeError: '_thread._local' object has no attribute 'value'
