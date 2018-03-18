# cw2vec
基于字符训练词向量,论文来源:https://github.com/ShelsonCao/cw2vec/blob/master/cw2vec.pdf


## 数据
训练数据来源: 采用ZAKER新闻半年的新闻数据,vocabulary_size为200000
汉字字符顺序解析: https://www.juhe.cn/docs/api/id/156


## 训练结果对比
采用word2vec跟cw2vec结果对比,测试集为wordsim-20,测试方法为Spearman's rank corrlation coefficient


## 目前进展
word2vec词向量已经训练完成,二维可视化结果为:https://github.com/Luka0612/cw2vec/blob/master/data/tsne.png

cw2vec处于清洗数据阶段