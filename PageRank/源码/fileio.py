import numpy as np   

class nodeGraph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, from_node, to_node):
        # 使用字典的方式构建邻接矩阵，优化稀疏矩阵
        if from_node not in self.graph:
            self.graph[from_node] = []
        self.graph[from_node].append(to_node)

    def print_graph(self):
        for node in self.graph:
            print(node, '->', self.graph[node])
    


def read_txt_file(file_path):
    """
    txt文件格式
    """
    graph = nodeGraph()
    max_node_id = 0
    with open(file_path, 'r') as file:
        for line in file:
            from_node, to_node = map(int, line.split())
            from_node = from_node
            to_node = to_node
            # 节点index从1开始，0没出和入节点，对收敛没有影响
            # 为了方便处理数组 直接从0开始
            graph.add_edge(from_node, to_node)
            max_node_id = max(max_node_id, from_node+1, to_node+1)
    #all_nodes = sorted(list(graph.graph.keys()))
    # print("所有节点的索引：", all_nodes)
    return graph, max_node_id

def save_txt_file(pagerank, file_path):
    with open(file_path, 'w') as file:
        for node_id, score in enumerate(pagerank, start=1):
            node_id = node_id
            file.write(f"{node_id} {score}\n")

def save_top_100_pagerank_nodes(pagerank, file_path, top_n=100):
    # 获取分数最高的节点索引
    top_indices = np.argsort(pagerank)[::-1][:top_n]
        
    with open(file_path, 'w') as file:
        for index in top_indices:
            file.write(f"{index} {pagerank[index]}\n")
