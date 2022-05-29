import pymysql
 
class MyDao:
    def __init__(self):
        pass
    
    def getEmps(self):
        ret = []
        db = pymysql.connect(host="49.50.174.207", user="noe", passwd="1234", db="noe", charset="utf8")
        curs = db.cursor()
        
        sql = "select * from noe";
        curs.execute(sql)
        
        rows = curs.fetchall()
        for e in rows:
            temp = {'title':e[0],'context':e[1]}
            ret.append(temp)
        
        db.commit()
        db.close()
        return ret
    
    def insEmp(self, title, context):
        db = pymysql.connect(host="49.50.174.207", user="noe", passwd="1234", db="noe", charset="utf8")
        curs = db.cursor()
        
        sql = '''insert into noe (title, context) values(%s,%s)'''
        curs.execute(sql,(title, context))
        db.commit()
        db.close()
    
    # def updEmp(self, empno, name, department,phone): 
    #     db = pymysql.connect(host='localhost', user='root', db='python', password='python', charset='utf8')
    #     curs = db.cursor()
        
    #     sql = "update emp set name=%s, department=%s, phone=%s where empno=%s"
    #     curs.execute(sql,(name, department, phone, empno))
    #     db.commit()
    #     db.close()
    # def delEmp(self, empno):
    #     db = pymysql.connect(host='localhost', user='root', db='python', password='python', charset='utf8')
    #     curs = db.cursor()
        
    #     sql = "delete from emp where empno=%s"
    #     curs.execute(sql,empno)
    #     db.commit()
    #     db.close()
 
if __name__ == '__main__':
    #MyEmpDao().insEmp('aaa', 'bb', 'cc', 'dd')
    #MyEmpDao().updEmp('aa', 'dd', 'dd', 'aa')
    #MyEmpDao().delEmp('aaa')
    noelist = MyDao().getEmps();
    print(noelist)