将Data.txt放在同目录下，运行pagerank.py即可，保证运行时的目录中有Data.txt

超参数需要麻烦手动调代码，在pagerank.py 147行调即可

初始固定为：teleport number = 0.85 | epsilon = 1e-8 | block_size=100

Pagerank_bs_100.txt为block-stripe的top 100

Pagerank_v_100.txt为普通pagerank的top 100



nx_pagerank仅用来对比nx包结果与输出结果