# !/usr/bin/python
# -*- coding:utf-8 -*-

import sys
import jieba
import jieba.posseg
from importlib import reload

if __name__ == "__main__":
    reload(sys)
    # sys.setdefaultencoding('utf-8')
    f = open('./24.novel.txt', encoding='utf-8')
    str = f.read()  # .decode('utf-8')
    f.close()

    seg = jieba.posseg.cut(str)
    for s in seg:
        print(s.word, s.flag, '|', )
        # print s.word, '|',
