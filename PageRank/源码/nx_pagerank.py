# 我们使用networkx的pagerank验证我们结果的正确性并做比较，
# 不直接使用networkx输出作为实验结果
import networkx as nx

def calculate_pagerank_from_txt(file_path):
    # 创建一个有向图
    G = nx.DiGraph()

    # 从txt文件中读取节点信息并添加到图中
    with open(file_path, 'r') as file:
        for line in file:
            # 分割每一行，获取 FromNodeID 和 ToNodeID
            from_node, to_node = map(int, line.strip().split())
            # 添加边
            G.add_edge(from_node, to_node)
    
    # 计算PageRank
    pagerank = nx.pagerank(G)

    # 打印分数最高的20个结果
    top_20 = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:20]
    for node, score in top_20:
        print(f"Node: {node}, PageRank: {score}")

if __name__ == "__main__":
    calculate_pagerank_from_txt("Data.txt")
