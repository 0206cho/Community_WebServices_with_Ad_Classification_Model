from flask import Flask, render_template, request	 # 플라스크 모듈 호출
import pymysql

from db import MyDao
from flask.json import jsonify

app = Flask(__name__) 		 # 플라스크 앱 생성

@app.route('/')				 # 기본('/') 웹주소로 요청이 오면 
def home():     			 # hello 함수 실행
    noelist = MyDao().getEmps();
    return render_template('home2.html',noelist=noelist)

# 글 추가
@app.route('/ins.ajax', methods=['POST'])
def ins_ajax():
    data = request.get_json()
    title = data['title']
    context = data['context']
    cnt = MyDao().insEmp(title, context)
    result = "success" if cnt==1 else "fail"
    return jsonify(result = result)

# 글쓰기 이동
@app.route('/home_write')
def home_write():
    return render_template('write.html');

# big html 이동
@app.route('/home_big')
def home_big():
    return render_template('big.html');
		
if __name__ == '__main__':	 # main함수
    app.run(debug=True, port=5000, host='0.0.0.0')

# ctlr + 5 -> localhost:5000 접속