import collections
import networkx as nx
import random
import numpy as np
import types
import cvxpy as cp
from mpi4py import MPI

"""
GraphProcessor
:description: GraphProcessor 设计用于预处理通信图，它规定了每个节点在每次迭代时的激活邻居
"""


class GraphProcessor(object):
    def __init__(self, base_graph, commBudget, rank, size, iterations, issubgraph):
        self.rank = rank # index of worker
        self.size = size # totoal number of workers
        self.comm = MPI.COMM_WORLD
        self.commBudget = commBudget # user defined budget

        if issubgraph:
            # if the base graph is already decomposed
            self.base_graph = self.getGraphFromSub(base_graph)
            self.subGraphs = base_graph
        else:
            # else: decompose the base graph
            # self.base_graph = base_graph
            self.base_graph = self.getGraphFromSub(base_graph)
            self.subGraphs = self.getSubGraphs()

        # get Laplacian matrices for subgraphs
        self.L_matrices = self.graphToLaplacian()

        # get neighbors' index
        self.neighbors_info = self.drawer()

    def getProbability(self):
        """ compute activation probabilities for subgraphs """
        raise NotImplemented

    def getAlpha(self):
        """ compute mixing weights """
        raise NotImplemented

    def set_flags(self, iterations):
        """ generate activation flags for each iteration """
        raise NotImplemented

    # 将一系列子图表示为边的列表，并使用这些边来构建一个 NetworkX 图
    def getGraphFromSub(self, subGraphs):
        G = nx.Graph()
        for edge in subGraphs:
            G.add_edges_from(edge)
        return G

    def getSubGraphs(self):
        """ 将基础图分解为匹配图 """
        G = self.base_graph
        subgraphs = list()

        # 首先尝试获得尽可能多的最大匹配
        for i in range(self.size-1):
            M1 = nx.max_weight_matching(G)  # 用于计算最大权重匹配的函数
            if nx.is_perfect_matching(G, M1):
                G.remove_edges_from(list(M1))
                subgraphs.append(list(M1))
            else:  # 如果图 G 包含多个最大权重匹配，并且它们的权重相同，那么具体的匹配结果可能会受到图的边顺序的影响。
                edge_list = list(G.edges)
                random.shuffle(edge_list)
                G.remove_edges_from(edge_list)
                G.add_edges_from(edge_list)
        print(f"list(G.edges)：{list(G.edges)}")
        print("--------------------------")
        # 使用贪婪算法分解剩余部分
        rpart = self.decomposition(list(G.edges))
        print(f"rpart：{rpart}")
        print("--------------------------")

        for sgraph in rpart:
            subgraphs.append(sgraph)

        return subgraphs

    # 将给定的子图列表 self.subGraphs 转换为拉普拉斯矩阵（Laplacian matrices）的列表
    def graphToLaplacian(self):
        L_matrices = list()
        for i, subgraph in enumerate(self.subGraphs):
            tmp_G = nx.Graph()
            tmp_G.add_edges_from(subgraph)
            L_matrices.append(nx.laplacian_matrix(tmp_G, list(range(self.size))).todense())

        return L_matrices

    def decomposition(self, graph):
        size = self.size

        node_degree = [[i, 0] for i in range(size)]  # 节点编号和节点的度
        node_to_node = [[] for i in range(size)]  # 用于存储每个节点与其邻居节点之间的连接关系。
        node_degree_dict = collections.defaultdict(int)  # 是一个字典，用于记录每个节点的度。
        node_set = set()   # 是一个集合，用于存储图中的所有节点。
        for edge in graph:  # 对节点的度和连接关系进行更新，并同时检查是否存在不合法的情况（重复的边或环路）。
            node1, node2 = edge[0], edge[1]
            node_degree[node1][1] += 1
            node_degree[node2][1] += 1
            if node1 in node_to_node[node2] or node2 in node_to_node[node1]:
                print("Invalid input graph! Double edge! ("+str(node1) +", "+ str(node2)+")")
                exit()
            if node1 == node2:
                print("Invalid input graph! Circle! ("+str(node1) +", "+ str(node2)+")")
                exit()
 
            node_to_node[node1].append(node2)
            node_to_node[node2].append(node1)
            node_degree_dict[node1] += 1
            node_degree_dict[node2] += 1
            node_set.add(node1)
            node_set.add(node2)

        node_degree = sorted(node_degree, key = lambda x: x[1])  # 按节点的度从高到低进行排序
        node_degree[:] = node_degree[::-1]  # 将结果反转，以便首先处理度最大的节点。
        subgraphs = []
        min_num = node_degree[0][1]  # 获取节点列表中度最大的节点的度，即 min_num。

        """
        循环遍历节点集合 node_set，并对每个节点查找与之相邻且度最大的节点（与其相邻且度最大的节点之间的边会被添加到子图中）。
        同时，更新节点的度、连接关系，从 node_set 中移除这些节点，然后将生成的子图添加到 subgraphs 列表中。
        """
        while node_set:
            subgraph = []
            for i in range(size):
                node1, node1_degree = node_degree[i]
                if node1 not in node_set:
                    continue
                for j in range(i+1, size):
                    node2, node2_degree = node_degree[j]
                    if node2 in node_set and node2 in node_to_node[node1]:  # ????
                        subgraph.append((node1, node2))
                        node_degree[j][1] -= 1
                        node_degree[i][1] -= 1
                        node_degree_dict[node1] -= 1
                        node_degree_dict[node2] -= 1
                        node_to_node[node1].remove(node2)
                        node_to_node[node2].remove(node1)
                        node_set.remove(node1)
                        node_set.remove(node2)
                        break
            subgraphs.append(subgraph)
            for node in node_degree_dict:
                if node_degree_dict[node] > 0:
                    node_set.add(node)
            node_degree = sorted(node_degree, key = lambda x: x[1])
            node_degree[:] = node_degree[::-1]
        return subgraphs

    # 将输入的图（表示为 self.subGraphs）表示为连接矩阵。连接矩阵是一个列表，其中每个元素是一个列表，用于表示节点之间的连接关系。
    def drawer(self):
        """
        input graph: list[list[tuples]]
                     [graph1, graph2,...]
                     graph: [edge1, edge2, ...]
                     edge: (node1, node2)
        output connect: matrix: [[]]
        """
        
        connect = []
        cnt = 1
        for graph in self.subGraphs:
            new_connect = [-1 for i in range(self.size)]
            for edge in graph:
                node1, node2 = edge[0], edge[1]
                if new_connect[node1] != -1 or new_connect[node2] != -1:
                    print("invalide graph! graph: "+str(cnt))
                    exit()
                new_connect[node1] = node2
                new_connect[node2] = node1
            # print(new_connect)
            connect.append(new_connect)
            cnt += 1
        return connect


class FixedProcessor(GraphProcessor):
    """ wrapper for fixed communication graph """

    def __init__(self, base_graph, commBudget, rank, size, iterations, issubgraph):
        super(FixedProcessor, self).__init__(base_graph, commBudget, rank, size, iterations, issubgraph)
        self.probabilities = self.getProbability()
        self.neighbor_weight = self.getAlpha()
        self.active_flags = self.set_flags(iterations + 1)

    def getProbability(self):
        """ activation probabilities are same for subgraphs """
        return self.commBudget 

    def getAlpha(self):
        """ 在这种情况下，α 有一个分析表达式 """
        L_base = np.zeros((self.size, self.size))
        for subLMatrix in self.L_matrices:
            L_base += subLMatrix
        w_b, _ = np.linalg.eig(L_base)
        lambdaList = list(sorted(w_b))
        if len(w_b) > 1:
            alpha = 2 / (lambdaList[1] + lambdaList[-1])

        return alpha

    def set_flags(self, iterations):
        """ warning: np.random.seed should be same across workers 
                     so that the activation flags are same

        """
        iterProb = np.random.binomial(1, self.probabilities, iterations)
        flags = list()
        idx = 0
        for prob in iterProb:
            if idx % 2 == 0:
                flags.append([0,1])
            else:
                flags.append([1,0])
            
            idx += 1
            # flags.append([prob for i in range(len(self.L_matrices))])

        return flags


class MatchaProcessor(GraphProcessor):
    """ Wrapper for MATCHA
        At each iteration, only a random subset of subgraphs are activated
    """

    def __init__(self, base_graph, commBudget, rank, size, iterations, issubgraph):
        super(MatchaProcessor, self).__init__(base_graph, commBudget, rank, size, iterations, issubgraph)
        self.probabilities = self.getProbability()          # 存储计算的激活概率
        self.neighbor_weight = self.getAlpha()              # 存储计算的邻居权重
        self.active_flags = self.set_flags(iterations + 1)  # 存储随机生成的激活标志，用于指示哪些子图在每次迭代中激活。
        self.rank = rank

    def getProbability(self):
        num_subgraphs = len(self.L_matrices)
        p = cp.Variable(num_subgraphs) # 定义向量变量
        L = p[0]*self.L_matrices[0]
        for i in range(num_subgraphs-1):
            L += p[i+1]*self.L_matrices[i+1]  # 线性组合矩阵 L
        eig = cp.lambda_sum_smallest(L, 2)  # 来计算 L 矩阵的两个最小特征值之和
        sum_p = p[0]
        for i in range(num_subgraphs-1):
            sum_p += p[i+1]

        # 激活概率的 cvx 优化
        obj_fn = eig
        constraint = [sum_p <= num_subgraphs*self.commBudget, p>=0, p<=1]  # 定义了一些约束条件
        problem = cp.Problem(cp.Maximize(obj_fn), constraint)  # 来计算 L 矩阵的两个最小特征值之和最大化

        # CVXOPT 是一个用于凸优化的 Python 库
        # cp.ROBUST_KKTSOLVER 作为 KKT 求解器。这个选项可以帮助提高求解的鲁棒性，使求解更加稳定。
        problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)

        # 求解
        tmp_p = p.value
        originActivationRatio = np.zeros((num_subgraphs))  # 于存储最优解的激活概率。
        for i, pval in enumerate(tmp_p):
            originActivationRatio[i] = np.real(float(pval))  # pval 转换为实数（确保是实数）

        return np.minimum(originActivationRatio, 1)  # 确保所有概率值都不超过 1。

    def getAlpha(self):
        num_subgraphs = len(self.L_matrices)  # 获取子图的数量
        num_nodes = self.size  # 获取节点的数量

        # 准备矩阵
        I = np.eye(num_nodes)  # 创建大小为 num_nodes x num_nodes 的单位矩阵
        J = np.ones((num_nodes, num_nodes)) / num_nodes  # 创建大小为 num_nodes x num_nodes 的均匀矩阵，每个元素都是 1/num_nodes

        mean_L = np.zeros((num_nodes, num_nodes))  # 创建一个全零的 num_nodes x num_nodes 矩阵
        var_L = np.zeros((num_nodes, num_nodes))  # 创建一个全零的 num_nodes x num_nodes 矩阵
        for i in range(num_subgraphs):
            val = self.probabilities[i]  # 获取概率值
            mean_L += self.L_matrices[i] * val  # 计算均值矩阵，每个元素等于对应子图的 L 矩阵元素乘以概率值
            var_L += self.L_matrices[i] * (1 - val) * val  # 计算方差矩阵，每个元素等于对应子图的 L 矩阵元素乘以 (1-概率值) 乘以概率值

        # SDP for mixing weight
        a = cp.Variable()  # 创建一个凸优化问题中的变量 a
        b = cp.Variable()  # 创建一个凸优化问题中的变量 b
        s = cp.Variable()  # 创建一个凸优化问题中的变量 s
        obj_fn = s  # 目标函数设置为最小化 s
        constraint = [(1 - s) * I - 2 * a * mean_L - J + b * (np.dot(mean_L, mean_L) + 2 * var_L) << 0, a >= 0, s >= 0,
                      b >= 0, cp.square(a) <= b]
        # 设置凸优化问题的约束条件。这些约束条件包括矩阵不等式约束 [(1-s)*I - 2*a*mean_L-J + b*(np.dot(mean_L,mean_L)+2*var_L) << 0]
        # 以及 a、s、b 变量的非负性约束和 a 的平方小于等于 b 的约束。

        problem = cp.Problem(cp.Minimize(obj_fn), constraint)  # 创建凸优化问题，最小化目标函数 s，满足约束条件 constraint
        problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)  # 求解凸优化问题，使用 CVXOPT 求解器和 ROBUST_KKTSOLVER 求解器

        return float(a.value)  # 返回变量 a 的值作为结果

    def set_flags(self, iterations):
        """ warning: np.random.seed（随机种子）应与各工作站相同，以便激活标志也相同

        """
        flags = list()
        # print(f"rank：{self.rank} -- {self.probabilities}")
        for i in range(len(self.L_matrices)): # 生成一个二项分布的随机值列表
            if np.isnan(self.probabilities[i]) or self.probabilities[i] < 0:
                self.probabilities[i] = 0
            flags.append(np.random.binomial(1, self.probabilities[i], iterations))

        return [list(x) for x in zip(*flags)]