from flask import Flask, render_template, request	 # 플라스크 모듈 호출

from db import MyDao
from flask.json import jsonify

# import torch
# from torch import nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import gluonnlp as nlp
# import numpy as np
# from tqdm.notebook import tqdm

# from kobert import get_tokenizer
# from kobert import get_pytorch_kobert_model

import pandas as pd

app = Flask(__name__) 		 # 플라스크 앱 생성

@app.route('/')				 # 기본('/') 웹주소로 요청이 오면 
def home():
    noelist = MyDao().getEmps();
    return render_template('home.html',noelist=noelist)

# 글 추가
@app.route('/ins.ajax', methods=['GET', 'POST'])
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
@app.route('/home_big', methods=['GET'])
def home_big():
    num = '%s' %request.args.get('num')
    noe = MyDao().getEmpss(num);
    ans = MyDao().getAnss(num);
    return render_template('big.html', noe=noe, ans = ans);


# 댓글 추가
@app.route('/ans_ins.ajax', methods=['GET', 'POST'])
def ans_ins_ajax():
    data = request.get_json()
    num = data['num']
    ans = data['ans']
    cnt = MyDao().insAns(num, ans)
    result = "success" if cnt==1 else "fail"
    return jsonify(result = result)

		
if __name__ == '__main__':	 # main함수
    app.run(debug=True, port=5000, host='0.0.0.0')

# ctlr + 5 -> localhost:5000 접속
