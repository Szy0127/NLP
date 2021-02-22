class Node:
    def __init__(self,flag=False):
        self.children = {}
        self.flag = flag
    def add_child(self,char,flag):
        child = self.children.get(char)
        if child is None:
            child = Node(flag)
            self.children[char] = child
        elif flag:
            child.flag = flag
        return child
        
class Trie(Node):
    def __init__(self):
        super().__init__()
        
    def add_word(self,word):
        state = self
        n = len(word)
        for i,char in enumerate(word):
            if i == n - 1:
                state = state.add_child(char,True)
            else:
                state = state.add_child(char,False)
    
    def find_word(self,word):
        state = self
        for char in word:
            if not char in state.children:
                return False
            else:
                state = state.children.get(char)
        return True

if __name__ == '__main__':
    trie = Trie()
    trie.add_word('自然')
    trie.add_word('自然人')
    trie.add_word('自然语言')
    trie.add_word('自语')
    assert trie.find_word('自然')
    assert trie.find_word('自然人')
    assert trie.find_word('自然语言')
    assert trie.find_word('自语')
