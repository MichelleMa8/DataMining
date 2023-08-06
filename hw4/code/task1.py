from pyspark import SparkConf, SparkContext
import sys, datetime
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from graphframes import GraphFrame

# build sc
conf = SparkConf().setAppName("task1").setMaster("local[*]")
sc = SparkContext(conf = conf)
spark = SparkSession.builder.appName("graph").getOrCreate()
sc.setLogLevel("WARN")
conf.set("spark.executor.memory", "4g")
conf.set("spark.driver.memory", "4g")

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    
    threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    data = sc.textFile(input_file_path).map(lambda x: x.split(','))
    firstline = data.first()
    user_bsn_rdd = data.filter(lambda x: x != firstline)

    # get all edges data
    user_bsngroup = user_bsn_rdd.groupByKey().mapValues(set)\
                        .filter(lambda x: len(x[1]) >= threshold).collectAsMap()
    # get all nodes data
    all_nodes = list(user_bsngroup.keys())
    
    # get filtered nodes data
    nodes_set = set()
    # store filtered edges data
    edges = []
    for i in range(len(all_nodes)):
        n1 = all_nodes[i]
        if n1 not in user_bsngroup:
            continue
        for j in range(i + 1, len(all_nodes)):
            n2 = all_nodes[j]
            if n2 in user_bsngroup:
                edge = len(user_bsngroup[n1] & user_bsngroup[n2])
                if edge >= threshold:
                    # add nodes
                    nodes_set.add(n1)
                    nodes_set.add(n2)
                    # undirected graph
                    edges.append([n1, n2])
                    edges.append([n2, n1])
    nodes = []
    for n in nodes_set:
        nodes.append([n])
        
    nodes_df = spark.createDataFrame(nodes,['id'])
    edges_df = spark.createDataFrame(edges, ['src','dst'])

    # generate graph
    graph = GraphFrame(nodes_df, edges_df)

    # train model
    res_df = graph.labelPropagation(maxIter = 5)
    result = res_df.rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(list)\
        .map(lambda x: sorted(x[1])).sortBy(lambda x: (len(x), x[0])).collect()
    
    # write into file
    file = open(output_file_path, 'w')
    for res_l in result:
        s = str(res_l)
        file.write(s[1:-1])
        file.write('\n')
    file.close()

    end_time = datetime.datetime.now()
    print((end_time - start_time).seconds)