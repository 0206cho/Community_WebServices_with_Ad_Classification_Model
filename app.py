import requests
from flask import Flask, render_template	 # 플라스크 모듈 호출

app = Flask(__name__) 		 # 플라스크 앱 생성

@app.route('/')				 # 기본('/') 웹주소로 요청이 오면 
def home():     			 # hello 함수 실행
    return render_template('home.html'); 

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

# debug : True이면 코드가 변경될때마다(저장포함) 서버가 자동으로 재실행된다

# ctlr + 5 -> localhost:5000 접속