from hmm import HMM
import re
'''
(一阶)隐马尔可夫模型：句子(观测序列) --> BMES(状态序列)
根据语料库获得模型的三个参数(四个标签之间的转换关系，汉字与四个标签的关系)
准确率不高(80%)  仅在OOV(词典没有的词)召回上领先词典分词
'''

def pre_process(sentence):
    if len(sentence)<2: #一个字的句子是来捣乱的
        return ''
    else:
        sentence = re.sub('[\s+\.\!\/_,$%^*(+\”、‘“”《》]+|[+—！，。？、~@#￥%…&*（）：:’]+',' ',sentence)#多个空格替换为一个
        sentence = re.sub(' +',' ',sentence)#多个空格替换为一个
        if sentence[0] == '“':
            sentence = sentence[1:]
        if sentence[0] == ' ':
            sentence = sentence[1:]
        if sentence[-1] != ' ': #分词以空格为标志 最后加一个
            sentence += ' '
    return sentence
                    
class CWS_HMM(HMM):
    def __init__(self,states,observations = []):
        super().__init__(states,observations)
        
    def train(self,path):
        with open(path,'r') as f:
            sentence = f.readline().strip()
            while sentence: 
                sentence = pre_process(sentence)
                if not sentence:
                    sentence = f.readline().strip()
                    continue
                if sentence[1] == ' ':  #先解决pi
                    cur_state = 'S'
                    index = 2
                else:
                    cur_state = 'B'
                    index = 1
                self.pi[cur_state] += 1
                for state in self.states:
                        if state == cur_state:
                            if sentence[0] in self.B[state]:
                                self.B[state][sentence[0]] += 1
                            else:
                                self.B[state][sentence[0]] = 1
                        else:
                            if not sentence[0] in self.B[state]:
                                self.B[state][sentence[0]] = 0

                pre_state = cur_state #状态序列的标注得看前面的状态和后面是不是空格
                i = index
                j = i+1
                while j<len(sentence):
                    if pre_state in ['B','M']:
                        if sentence[j] == ' ':
                            cur_state = 'E'
                            i += 2
                        else:
                            cur_state = 'M'
                            i += 1
                    else:
                        if sentence[j] == ' ':
                            cur_state = 'S'
                            i += 2
                        else:
                            cur_state = 'B'
                            i += 1

                    for state in self.states:
                        if state == cur_state:
                            if sentence[j-1] in self.B[state]:
                                self.B[state][sentence[j-1]] += 1
                            else:
                                self.B[state][sentence[j-1]] = 1
                        else:
                            if not sentence[j-1] in self.B[state]:
                                self.B[state][sentence[j-1]] = 0
                
                    j = i+1
                    self.A[pre_state][cur_state] += 1
                    pre_state = cur_state
                
                sentence = f.readline().strip()
               

        s = sum(self.pi.values())#频率统计完成 开始计算概率
        for state in self.pi:
            self.pi[state]/=s
        for statei in self.A:
            s = sum(self.A[statei].values())
            if s!=0:
                for statej in self.A[statei]:
                    self.A[statei][statej]/=s
        for state in self.B:
            s = sum(self.B[statei].values())
            if s!=0:
                for char in self.B[state]:
                    self.B[state][char]/=s
            
        
    def predict(self,sentence):
        x = [char for char in sentence]#需传入观测序列
        y = super().predict(x)  #返回states序列 即BMES
        predict = []
        buffer = ''
        for char,state in zip(x,y):
            if state in ['B','S'] and buffer != '':
                predict.append(buffer)
                buffer = ''
            buffer += char
        if buffer:
            predict.append(buffer)

        for word in predict:
            print(word,end=' ')
        
        
states = ('B','M','E','S')     
segment = CWS_HMM(states)
segment.train('msr_training.txt')
segment.predict('研究生命起源')
