# cw2vec
基于字符训练词向量,论文来源:https://github.com/ShelsonCao/cw2vec/blob/master/cw2vec.pdf

同时也有了解到其他有关于字符训练的论文：https://arxiv.org/pdf/1704.04859.pdf?

以及一些优化想法

## 数据
训练数据来源: 采用ZAKER新闻半年的新闻数据,vocabulary_size为200000

汉字字符顺序解析: https://www.juhe.cn/docs/api/id/156


## 训练结果对比
采用word2vec跟cw2vec结果对比,测试集为wordsim-240,测试方法为Spearman's rank corrlation coefficient


## 目前进展
word2vec词向量已经训练完成,二维可视化结果为:https://github.com/Luka0612/cw2vec/blob/master/data/tsne.png

cw2vec数据清洗完成,处于训练阶段

## cw2vec
model_train_sum_sroke训练时间很长,还涉及到超参数的优化等,目前采用的优化算法是tf.train.GradientDescentOptimizer(),学习率是0.1
由于还涉及到与word2vec还有Character-level的比较,尽量使得参数比较相近

## 其他对比论文
对词语进行可视化CNN训练：https://arxiv.org/pdf/1704.04859.pdf?

## 优化方向
cw2word与Learning Character-level Compositionality with Visual Features感觉都有缺陷，没有考虑字符的书写顺序，可以考虑在字符到词向量层增加一层RNN进行字符书写顺序信息的获取。