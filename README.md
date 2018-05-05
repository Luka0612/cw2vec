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

cw2vec训练&调参会花费较长时间,这个时间段会考虑先实现Character-level to vec
## cw2vec
model_train_sum_sroke训练时间很长,还涉及到超参数的优化等,目前采用的优化算法是tf.train.GradientDescentOptimizer(),学习率是0.1。
由于还涉及到与word2vec还有Character-level的比较,尽量使得参数比较相近

由于我的渣电脑训练的比较慢,而且涉及200000vocabulary还有半年的ZAKER新闻数据,训练还需要段时间

MMP，不小心关掉训练了。。。。得重新开始训练，哭嘤嘤，还是等服务器吧

## Character-level2vec
Learning Character-level Compositionality with Visual Features是每个字为单位进行CNN处理后输入RNN预测标题分类，
由于我们是只训练词向量且为了具有可比性采用跟Word2vec相同的架构，所以我对其进行了些修改，采用以词作为单位进行CNN处理，CNN结果预测上下文（类似word2vec）训练词向量。
暂时还不知道以词作为单位对CNN效果是否不够好（因为词的长度是不同的，有两字词，四字词等）。
如果以字作为单位的话，可以对词的每个字进行CNN处理，然后进行RNN形成词向量，再根据上下文进行训练词向量
## stroke-rnn2vec
个人的一些想法，是否可以将字符（如词“大人”的字符13434）进行RNN处理，从而考虑字符的书写顺序，然后结合上下文训练词向量

## 其他对比论文
对词语进行可视化CNN训练：https://arxiv.org/pdf/1704.04859.pdf?

## 优化方向
cw2word与Learning Character-level Compositionality with Visual Features感觉都有缺陷，没有考虑字符的书写顺序，可以考虑在字符到词向量层增加一层RNN进行字符书写顺序信息的获取。