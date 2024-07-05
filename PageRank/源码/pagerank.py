from fileio import nodeGraph, read_txt_file, save_txt_file, save_top_100_pagerank_nodes
import numpy as np
import time
def normalize_columns(matrix):
    # 按列即每个节点的输出链接到的节点概率值进行归一化
    col_sums = matrix.sum(axis=0)
    all_zeros_cols = col_sums == 0
    col_sums[all_zeros_cols] = 1
    return matrix / col_sums[np.newaxis, :]

def construct_link_matrix(graph, num_nodes):
    link_matrix = np.zeros((num_nodes, num_nodes))

    # 构建基础链接节点
    for from_node, to_nodes in graph.graph.items():
        for to_node in to_nodes:
            link_matrix[to_node][from_node] = 1

    # 处理"Dead End"节点，将其等概率链接到所有其他节点
    dead_end_nodes = [node for node in range(0, num_nodes) if node not in graph.graph]
    for node in dead_end_nodes:
        link_matrix[:, node] = 1
    # 归一化
    link_matrix = normalize_columns(link_matrix)
    return link_matrix

def construct_part_link_matrix(graph, num_nodes, ina, inb):
    # 直接从graph中构建部分链接矩阵即stripe
    # index范围为ina到inb-1 
    link_matrix = np.zeros((inb-ina, num_nodes))

    # 构建基础链接节点
    for from_node, to_nodes in graph.graph.items():
        for to_node in to_nodes:
            # 如果destination在范围中
            if to_node in range(ina, inb):
                link_matrix[to_node-ina][from_node] = 1

    # 处理"Dead End"节点，将其等概率链接到所有其他节点
    dead_end_nodes = [node for node in range(0, num_nodes) if node not in graph.graph]
    for node in dead_end_nodes:
        link_matrix[:, node] = 1
    # 归一化
    link_matrix = normalize_columns(link_matrix)
    return link_matrix

def simulate_store_link_matrix(graph,num_nodes,block_size):
    # 构建一个M的stripe的列表
    M_list = []
    for block_start in range(0, num_nodes, block_size):
        block_end = min(block_start + block_size, num_nodes)
        # 如有需要，可以将block保存到磁盘，对block迭代再读取
        # 由于要求可以在内存里对block运算，为了代码实现的简洁，这里
        # 用一个list存储在内存中
        M_list.append(construct_part_link_matrix(graph, num_nodes, block_start, block_end))
    return M_list

def simulate_store_pagerank(num_nodes,block_size):
    # 构建分块的pagerank，初始化为1/N
    pagerank = np.ones(num_nodes) / num_nodes
    r_list = []
    for block_start in range(0, num_nodes, block_size):
        block_end = min(block_start + block_size, num_nodes)
        # 如有需要，可以将block保存到磁盘，对block迭代再读取
        # 由于要求可以在内存里对block运算，为了代码实现的简洁，这里
        # 用一个list存储在内存中
        r_list.append(pagerank[block_start:block_end])
    return r_list

def compute_pagerank(link_matrix, tp=0.85, epsilon=1e-8, max_iterations=1000):
    # naive pagerank + spider trap/dead ends handling
    num_nodes = link_matrix.shape[0]
    pagerank = np.ones(num_nodes) / num_nodes

    # for _ in range(max_iterations):
    #     new_pagerank = np.dot(link_matrix, pagerank)
    #     if np.linalg.norm(new_pagerank - pagerank) < epsilon:
    #         return new_pagerank
    #     pagerank = new_pagerank

    for _ in range(max_iterations):
        # 带阻尼因子的random teleport解决spider trap
        new_pagerank = (1 - tp) / num_nodes + \
                              tp * np.dot(link_matrix, pagerank)
        # 欧式距离与阈值判断收敛
        if np.linalg.norm(new_pagerank - pagerank) < epsilon:
            return new_pagerank
        pagerank = new_pagerank

    return pagerank

def block_stripe_pagerank(M_list, r_list, tp=0.85, epsilon=1e-8, max_iterations=1000):
    num_nodes = M_list[0].shape[1]
    r_old =  np.concatenate(r_list, axis=0)
    
    # Block-Stripe更新算法
    for _ in range(max_iterations):
        new_pagerank = np.zeros(num_nodes)
        block_idx = 0
        # 每一行存储destination=index的节点
        # 然后和r_old点乘即可
        for M_cur in M_list:
            # 带阻尼因子的random teleport解决spider trap
            r_cur = (1 - tp) / num_nodes + \
                              tp * np.dot(M_cur, r_old)
            # 分块存储r
            r_list[block_idx] = r_cur
            block_idx += 1
        r_new = np.concatenate(r_list, axis=0)
        # 欧式距离与阈值判断收敛
        if np.linalg.norm(r_new - r_old) < epsilon:
            return r_new
        r_old = r_new
    
    return pagerank

def block_stripe_pagerank_read(num_nodes, block_size, r_list, tp=0.85, epsilon=1e-8, max_iterations=1000):
    # 每次都从邻接矩阵读取M的stripe
    # 耗时太长，main函数中没用上，仅供参考
    r_old =  np.concatenate(r_list, axis=0)
    
    # Block-Stripe更新算法
    for _ in range(max_iterations):
        new_pagerank = np.zeros(num_nodes)
        block_idx = 0
        for block_start in range(0, num_nodes, block_size):
            block_end = min(block_start + block_size, num_nodes)
            M_cur = construct_part_link_matrix(graph, num_nodes, block_start, block_end)
            r_cur = (1 - tp) / num_nodes + \
                              tp * np.dot(M_cur, r_old)
            r_list[block_idx] = r_cur
            block_idx += 1
        r_new = np.concatenate(r_list, axis=0)
        if np.linalg.norm(r_new - r_old) < epsilon:
            return r_new
        r_old = r_new
    
    return pagerank


if __name__ == "__main__":
    # 示例用法
    file_path = "./Data.txt"  # 替换成你的文件路径
    graph, num_nodes = read_txt_file(file_path)

    # hyper params
    tp = 0.85
    block_size = 100
    epsilon=1e-8
    start = time.time()
    end = time.time()

    # -- vanilla pagerank
    start_v = time.time()
    link_matrix = construct_link_matrix(graph, num_nodes)
    pagerank = compute_pagerank(link_matrix, tp, epsilon)
    end_v = time.time()
    print(f"Vanilla PageRank Done, Time:{end_v-start_v}s.")
    save_top_100_pagerank_nodes(pagerank, "./Pagerank_v_100.txt")
    

    # -- block stripe pagerank
    start_bs = time.time()
    M_list = simulate_store_link_matrix(graph, num_nodes, block_size)
    r_list = simulate_store_pagerank(num_nodes, block_size)
    pagerank = block_stripe_pagerank(M_list, r_list, tp, epsilon)
    end_bs = time.time()
    print(f"Block-stripe PageRank Done, Time:{end_v-start_v}s.")
    save_top_100_pagerank_nodes(pagerank, "./Pagerank_bs_100.txt")