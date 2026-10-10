from nltk.corpus import wordnet
# print(nltk.data.path)

# nltk.download('wordnet', download_dir=r'D:\nltk_data')

print(wordnet.synsets('car'))       # 输出 car 的所有同义词簇

car1 = wordnet.synset('car.n.01')           # 同义词簇，在 WordNet 中，词簇以 Synset 对象的形式存在。
print(car1.definition()) # type:ignore      # 该同义词簇的释义
print(car1.lemma_names()) # type:ignore     # 该同义词簇中包含的单词

# car2 = wordnet.synset('car.n.02')
# print(car2.definition()) # type:ignore

# car3 = wordnet.synset('car.n.03')
# print(car3.definition()) # type:ignore

# car4 = wordnet.synset('car.n.04')
# print(car4.definition()) # type:ignore

# car5 = wordnet.synset('cable_car.n.01')
# print(car5.definition()) # type:ignore


# 查看 car.n.01 词簇和其他词簇在词义上的上下位关系：
# WordNet 的网状结构是由词簇（Synset）作为节点连接起来的，而不是由单词（Word form）直接连接 ！ ！ ！
print(car1.hypernym_paths()[0]) # type:ignore


# 基于 WordNet 的词簇语义相似度计算：

car1 = wordnet.synset('car.n.01')
novel1 = wordnet.synset('novel.n.01')
dog1 = wordnet.synset('dog.n.01')
motorcycle1 = wordnet.synset('motorcycle.n.01')

# path_similarity()方法： 0~1的实数，越大就越相似
print(car1.path_similarity(novel1))         # type:ignore
print(car1.path_similarity(dog1))           # type:ignore
print(car1.path_similarity(motorcycle1))    # type:ignore
