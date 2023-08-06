from pyspark import SparkConf, SparkContext
import sys, datetime, itertools
from collections import defaultdict, deque
from operator import add

class TreeNode:
    def __init__(self, name):
        self.name = name
        self.val = 1
        self.depth = 0
        self.parent = [] # [parent node name]
        self.children = [] # [child node name]

def build_tree(root, test_edge):
    '''
    up-bottom to build the tree
    '''
    new_queue = {root.name: root}
    level_node = {0: [root.name]}
    depth = 0
    tree = {} # node name: node
    while new_queue:
        queue = new_queue
        new_queue = {}
        level = []
        depth += 1
        for name, node in queue.items():           
            if name in tree:
                continue
            tree[name] = node
            
            if name in test_edge:
                children_names = test_edge[name]

                for child_name in children_names:                       
                    if child_name not in tree and child_name not in queue:  
                        if child_name not in new_queue:
                            child_node = TreeNode(child_name)
                            new_queue[child_node.name] = child_node
                            level.append(child_node.name)
                        else:
                            child_node = new_queue[child_name]
                    
                        # update this level's children
                        node.children.append(child_node.name)
                        # update next level's parent
                        child_node.parent.append(name)
                        #print(name, child_name, tree[name].children)


                    
        if level:
            level_node[depth] = level
    return tree, level_node

def cal_betweenness(tree_dict, level_node):
    '''
    bottom-up the tree to calculate the betweenness
    '''
    betweenness = {}
    key = max(level_node.keys())

    while key >= 0:
        # ['b', 'g']
        node_list = level_node[key]
        key -= 1
        # g
        for node_name in node_list:
            # update node.val
            node = tree_dict[node_name]
            edge = 0
            for child_name in node.children:
                child_node = tree_dict[child_name]
                edge += betweenness[tuple(sorted([node_name, child_name]))]
                    
            node.val += edge
            
            # record betweenness
            parent_list = node.parent
            if len(parent_list) == 1:
                betweenness[tuple(sorted([node.name, parent_list[0]]))] = node.val
                             
            else:
                total = 0
                node_count = {}
                # [d, f]
                for parent_name in parent_list:
                    # d
                    parent_node = tree_dict[parent_name]
                    node_count[parent_name] = len(parent_node.parent)
                    total += len(parent_node.parent)
                for parent_name in parent_list:
                    node_count[parent_name] /= total
                    
                    betweenness[tuple(sorted([node.name, parent_name]))] = node_count[parent_name]  * node.val
            
    return betweenness

def multi_node_betweenness(root, test_edge):
    tree_dict, level_node = build_tree(root, test_edge)
    betweenness = cal_betweenness(tree_dict, level_node)
    return list(betweenness.items())

def find_one_cluster(graph, node, visited):
    cluster = set()
    def dfs_cluster(node, visited):
        if node in cluster:
            return
        cluster.add(node)
        for n in graph[node]:
            dfs_cluster(n, visited)
    dfs_cluster(node, visited)
    return cluster

def get_all_clusters(graph):
    cluster_all = {}
    i = 0
    visited = set()
    for node in graph:
        if node not in visited:
            cluster = find_one_cluster(graph, node, visited)
            visited |= cluster
            cluster_all[i] = cluster
            i += 1
    return cluster_all    

def cal_modularity(cluster_all, ori_graph):
    modularity = 0
    for nodes in cluster_all.values():
        degrees = 0
        A = 0
        for pair in itertools.combinations(nodes, 2):
            if pair[0] in ori_graph[pair[1]]:
                A += 1
            if pair[0] in K_map and pair[1] in K_map:
                degrees += K_map[pair[0]] * K_map[pair[1]]
        modularity += A - degrees / (2 * n_edge)
    return modularity / (2 * n_edge)

def Girvan_Newman(nodes_rdd, graph):
    ori_graph = graph
    stop_n = len(ori_graph)
    # get existing clusters
    cluster_all = get_all_clusters(graph)
    res_cluster = cluster_all
    # calculate modularity
    max_modularity = cal_modularity(cluster_all, ori_graph)

    early_stopping = 20
    
    while len(cluster_all) != stop_n:
        # calculate betweenness:
        betweenness_rdd = nodes_rdd.flatMap(lambda x: multi_node_betweenness(TreeNode(x), graph))\
                    .reduceByKey(add).map(lambda x: (x[0], x[1] / 2))\
                    .sortBy(lambda x: (-x[1], x[0][0]))
        
        # pick out the pair nodes with the max betweenness
        max_betweenness = betweenness_rdd.first()[1]
        max_betweenness_pair = betweenness_rdd.filter(lambda x: x[1] == max_betweenness).collect()
        # cut edges
        for pair in max_betweenness_pair:
            node1, node2 = pair[0][0], pair[0][1]
            graph[node1].remove(node2)
            graph[node2].remove(node1)
        # get clusters
        cluster_all = get_all_clusters(graph)
        # calculate modularity
        modularity = cal_modularity(cluster_all, ori_graph)
        if modularity > max_modularity:
            max_modularity = modularity
            res_cluster = cluster_all
            early_stopping = 20
        else:
            early_stopping -= 1
            if early_stopping == 0:
                break
                
    return max_modularity, res_cluster

if __name__ == '__main__':
    start_time = datetime.datetime.now()

    # build sc
    conf = SparkConf().setAppName("task2").setMaster("local[*]")
    sc = SparkContext(conf = conf)
    sc.setLogLevel("WARN")
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.driver.memory", "4g")

    threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    betweenness_output_path = sys.argv[3]
    community_output_path = sys.argv[4]

    data = sc.textFile(input_file_path).map(lambda x: x.split(','))
    firstline = data.first()
    user_bsn_rdd = data.filter(lambda x: x != firstline)

    # get nodes data
    nodes_rdd = user_bsn_rdd.map(lambda x: x[0]).distinct()
    nodes = nodes_rdd.collect()
    n_node = len(nodes)
    # get user_business group data
    user_bsngroup = user_bsn_rdd.groupByKey().mapValues(set).collectAsMap()

    # create graph
    graph = defaultdict(set)
    single_node_cluster = []
    n_edge = 0
    for i in range(len(nodes)):
        n1 = nodes[i]
        if n1 not in user_bsngroup:
            continue
        for j in range(i + 1, len(nodes)):
            n2 = nodes[j]
            if n2 not in user_bsngroup:
                continue
            edge = len(user_bsngroup[n1] & user_bsngroup[n2])
            if edge >= threshold:
                # undirected graph
                graph[n1].add(n2)
                graph[n2].add(n1)
                n_edge += 1
            
    K_map = {key: len(val) for key, val in graph.items()}
                
    # 1. Betweenness Calculation
    betweenness_rdd = nodes_rdd.flatMap(lambda x: multi_node_betweenness(TreeNode(x), graph))\
                        .reduceByKey(add).map(lambda x: (x[0], round(x[1] / 2, 5)))\
                        .sortBy(lambda x: (-x[1], x[0][0]))
    betweenness_res = betweenness_rdd.collect()
    # write betweenness results
    betweenness_file = open(betweenness_output_path, 'w')
    for pair, betweenness in betweenness_res:
        betweenness_file.write(str(pair) + ',' + str(betweenness) + '\n')
    betweenness_file.close()


    # 2. Community Detection
    max_modularity, res_cluster = Girvan_Newman(nodes_rdd, graph)
    res_community = defaultdict(list)
    for nodes in res_cluster.values():
        nodes_list = sorted(list(nodes))
        res_community[len(nodes_list)].append(nodes_list)
    # write betweenness results
    community_file = open(community_output_path, 'w')
    for key in sorted(res_community.keys()):
        if len(res_community[key]) > 1:
            res = sorted(res_community[key], key = lambda x: x[0])
            for r in res:
                community_file.write(str(sorted(r))[1:-1] + '\n')
        else:
            community_file.write(str(res_community[key])[2:-2] + '\n')
    community_file.close()

    end_time = datetime.datetime.now()
    print((end_time - start_time).seconds)