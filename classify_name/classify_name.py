import numpy as np

class PerceptronNameGenderClassifier():
    def __init__(self):
        self.X = [] #数据集特征向量
        self.Y = [] #数据集标注
        self.character_dict = {}    #特征为所有字符 此字典作用为快速索引
        self.w = [] #权重向量 
        
    def name2vector(self,name):
            vector = np.zeros(len(self.character_dict)+1)
            for character in name[1:]:#去掉姓氏
                try:
                    vector[self.character_dict[character]] = 1
                except:
                    continue
            return vector

    def sign(self,x):
        return 1 if x>=0 else -1

    def predict(self,x):    #根据特征向量输出1或1
        return self.sign(np.dot(self.w,x))

    def classify(self,name):     #根据名字输出性别
        y = self.predict(self.name2vector(name))
        gender = '男' if y == 1 else '女'
        print(name,gender)

    def load_data(self,path = 'train.csv'):
        print('开始处理数据')
        X = []  # 名字
        Y = []  # 性别
        with open(path,'rb') as f:  
            text_new = f.readline().decode('utf-8')
            while text_new:
                name,gender = text_new.split(',')
                X.append(name)
                Y.append(gender[0])
                text_new = f.readline().decode('utf-8')
                
        count = 0
        for name in X:
            for character in name[1:]:  #去掉姓
                if not character in self.character_dict:
                    self.character_dict[character] = count
                    count += 1
        
        for i in range(len(X)):     
            self.X.append(self.name2vector(X[i]))
            Yi = 1 if Y[i] == '男' else -1
            self.Y.append(Yi)

    def test(self,path = 'test.csv'):
        X = []  # 名字
        Y = []  # 性别
        with open(path,'rb') as f:  
            text_new = f.readline().decode('utf-8')
            while text_new:
                name,gender = text_new.split(',')
                X.append(name)
                Y.append(gender[0])
                text_new = f.readline().decode('utf-8')

        count = 0        
        for i in range(len(X)):
            x = self.name2vector(X[i])
            y = 1 if Y[i] == '男' else -1
            if y == self.predict(x):
                count += 1
        print('测试完成，正确率为{}%'.format(count/len(X)*100))
            
    def score(self):
        count = 0
        for index in range(len(self.X)):
            x = self.X[index]
            y = self.Y[index]
            if y == self.predict(x):
                count += 1
        return count/len(self.X)
    
    def train(self,epoch = 50):      
        self.w = np.zeros(len(self.X[0]))
        print('训练开始')
        for i in range(epoch):
            for index in range(len(self.X)):
                x = self.X[index]
                y = self.Y[index]
                if not y == self.predict(x):
                    self.w += y * x #感知机算法的迭代公式
                
model = PerceptronNameGenderClassifier()
model.load_data()
model.train()
print(model.score())
#model.test()
for name in ['霍建华','刘诗诗','王建国','李雪琴']:
    model.classify(name)
