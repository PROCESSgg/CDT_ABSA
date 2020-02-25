import pickle   #将Python对象序列化成一个字节流以便将它保存到一个文件、存储到数据库或者通过网络传输它。
#注意，这里是保存这个Python对象，你把这个读回来时候，还有这个对象的所有属性方法，这里是将Vocab这个对象序列化保存了。
import tqdm
from collections import Counter


class Vocab(object):
    def __init__(self, counter, specials=['<pad>', '<unk>']):
        #self.itos是按词频降序排序的列表
        #self.soti是itos单词索引的字典
        self.pad_index = 0
        self.unk_index = 1
        counter = counter.copy()
        self.itos = list(specials)
        for tok in specials:
            del counter[tok]
        
        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}


    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True
    ##这种写法相当于对Python内置函数在这个类上的一个重写，在这个类上使用len()就会调用这个方法。相当于这是对Python内置函数重写的一种方法。这属于Python的magic method
    def __len__(self):
        return len(self.itos)

    def extend(self, v):
        words = v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
        return self

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

