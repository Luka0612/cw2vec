# cw2vec
基于字符训练词向量,论文来源:https://github.com/ShelsonCao/cw2vec/blob/master/cw2vec.pdf

同时也有了解到其他有关于字符训练的论文：https://arxiv.org/pdf/1704.04859.pdf?

以及一些优化想法

## 数据
[训练数据](https://share.weiyun.com/5QUinSe)来源: 采用ZAKER新闻半年的新闻数据,vocabulary_size为200000

[分词、index处理后的数据](https://share.weiyun.com/5SlrabM)

[汉字字符顺序解析](https://www.juhe.cn/docs/api/id/156)


## 训练结果对比
采用word2vec跟cw2vec结果对比,测试集为[wordsim-240](https://github.com/Luka0612/cw2vec/tree/master/data/words-240%E5%8F%91%E5%B8%83),测试方法为Spearman's rank corrlation coefficient

暂由于其他项目还没空对模型进行对比,所以将模型&训练数据上传到微云,有兴趣的朋友可以做下测试,cw2vec优势:
* cw2vec使得对没有在训练数据中出现的单词（词汇表外的单词）计算单词的表征成为了可能
* 对于近义词间可以考虑字符的相似性如学校&学生都存在"学"相同字符


## 目前进展
word2vec词向量已经训练完成,[二维可视化结果](https://github.com/Luka0612/cw2vec/blob/master/data/tsne.png)

cw2vec数据清洗完成,训练完成，抽空会对训练的结果进行对比，同时将模型分享出来

[word2vec&cw2vec](https://share.weiyun.com/529Oiwc)模型已经上传微云

[导入模型可以参考下这个](https://github.com/Luka0612/cw2vec/blob/master/lib/load_model/load_model.py)

[Character-level to vec](https://github.com/Luka0612/cw2vec/tree/master/lib/train/cl2vec)已经实现

## cw2vec
model_train_sum_sroke训练时间很长,还涉及到超参数的优化等,目前采用的优化算法是tf.train.GradientDescentOptimizer(),学习率是0.1。

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
17~18年涌现很多关于n元字符的模型(解决新词的表征)训练,包括:
* [fastText](https://github.com/facebookresearch/fastText)
* [ELMo](https://arxiv.org/abs/1802.05365)