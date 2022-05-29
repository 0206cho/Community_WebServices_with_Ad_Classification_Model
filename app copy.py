import requests
from flask import Flask, render_template	 # 플라스크 모듈 호출
import pymysql

app = Flask(__name__) 		 # 플라스크 앱 생성

# database 접근
db = pymysql.connect(host="49.50.174.207", user="noe", passwd="1234", db="noe", charset="utf8")
# database 사용하기 위해 cursor 세팅
cur = db.cursor()

sql = "SELECT * from noe"
cur.execute(sql)

data_list = cur.fetchall()

@app.route('/')				 # 기본('/') 웹주소로 요청이 오면 
def home():     			 # hello 함수 실행
    # return render_template('home.html'); 
    sql = "SELECT * from noe"
    cur.execute(sql)
    data_list = cur.fetchall()

    return render_template('home.html',data_list=data_list)

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

    



# print(data_list[0])
# print(data_list[1])
# print(data_list[2])