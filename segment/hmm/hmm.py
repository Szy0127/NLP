import numpy as np
states = ('Healthy', 'Fever')
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
transition_probability = {}
transition_probability['Healthy'] = {'Healthy': 0.7, 'Fever': 0.3}
transition_probability['Fever'] = {'Healthy': 0.4, 'Fever': 0.6}
emission_probability = {}
emission_probability['Healthy'] = {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1}
emission_probability['Fever'] = {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
observations = ('normal', 'cold', 'dizzy')

class HMM():
    def __init__(self,states,observations = []):
        self.states = states
        self.state_amount = len(states)
        self.observations = observations
        self.pi = {}
        self.A = {}
        self.B = {}
        
        transition_probability = {state:0 for state in states}
        for state in states:    #先初始化为0 然后在train中计算频率 最后得到概率
            self.pi[state] = 0
            self.A[state] = transition_probability.copy()
            self.B[state] = {}
            
    def train(self):
        '''
        应该是根据数据集统计(极大似然估计)
        这边直接给出
        '''
        self.pi = start_probability
        self.A = transition_probability
        self.B = emission_probability

    def viterbi(self,state_map):
        distance = np.zeros(state_map.shape[:2])    #从start到[t][state]的最短路(概率最大)
        path = -np.ones(state_map.shape[:2])        #记录最短路的节点
        state_list = []         #最后的最短路
        for state in range(self.state_amount):
            distance[1][state] = state_map[0][0][state]
        for t in range(2,state_map.shape[0]):
            for state in range(self.state_amount):
                dist_list = [distance[t-1][statei]+state_map[t-1][statei][state] for statei in range(self.state_amount)]
                distance[t][state] = min(dist_list)
                path[t][state] = dist_list.index(distance[t][state])
                
        dist_end = [distance[state_map.shape[0]-1][state]for state in range(self.state_amount)]
        ans = min(dist_end)
        path_end = dist_end.index(ans)
        state = int(path_end)

        state_list.append(state)
        for t in range(state_map.shape[0]-1,1,-1):#反向得出路径
            state = int(path[t][state])
            state_list.append(state)

        state_list = state_list[::-1]
        return state_list
        
    def predict(self,x):
        day = len(x)
        state_map = np.zeros((day+1,self.state_amount,self.state_amount)) #[t][s1][s2]表示第t天的s1状态转移到第t+1天的s2状态
        for state in range(self.state_amount):
            if not x[0] in self.B[self.states[state]]:#字没有在语料库中出现过 为了不报错 设置为0
                state_map[0][0][state] = 0
            else:
                state_map[0][0][state] = self.pi[self.states[state]]* self.B[self.states[state]][x[0]]
            if state_map[0][0][state] == 0:
                state_map[0][0][state] = 1e-100
            state_map[0][0][state] = -np.log(state_map[0][0][state])#防止在过程中概率太小变成NaN无法进行    
        for t in range(1,day):  #天数索引为1,2,...n 状态索引为0,1,2...t
            for statei in range(self.state_amount):
                for statej in range(self.state_amount):
                    if not x[t] in self.B[self.states[statej]]:
                        state_map[t][statei][statej] = 0
                    else:#这边x[t]是第t+1天的观察态 索引的问题
                        state_map[t][statei][statej] = self.A[self.states[statei]][self.states[statej]] * self.B[self.states[statej]][x[t]]
                    if state_map[t][statei][statej]  == 0 :
                        state_map[t][statei][statej] = 1e-100
                    state_map[t][statei][statej] = -np.log(state_map[t][statei][statej])
        state_list = self.viterbi(state_map)
        predict_states = [self.states[state] for state in state_list]
        return predict_states
        
        
if __name__ == '__main__':
    model = HMM(states,observations)
    model.train()
    print(model.predict(['normal','cold','dizzy']))#不能少于两天 否则报错
        
