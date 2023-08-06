from pyspark import SparkConf, SparkContext
import sys, datetime
from operator import add
from collections import defaultdict
from sklearn.cluster import KMeans

class DS_CS:
    def __init__(self, name, data_id):
        self.name = name
        self.data_id = set(data_id)
        self.n = len(data_id)
        n_feature = len(data_map[data_id[0]])
        self.sum_feature = [0] * n_feature
        self.sumsq_feature = [0] *  n_feature
        for data in data_id:
            feature = data_map[data]
            for i in range(n_feature):
                self.sum_feature[i] += feature[i]
                self.sumsq_feature[i] += feature[i] ** 2
        
    def get_centroid(self):
        return [(val / self.n) for val in self.sum_feature]
    
    def cal_var(self):
        return [(self.sumsq_feature[i]/self.n - (self.sum_feature[i]/self.n)**2) for i in range(len(self.sum_feature))]      
    
    def add_point(self, index):
        self.n += 1
        self.data_id.add(index)
        feature = data_map[index]
        for i in range(len(feature)):
            self.sum_feature[i] += feature[i]
            self.sumsq_feature[i] += feature[i] ** 2

def get_rs(model_rs, X_idx_pre):
    # count cluster
    labels = model_rs.labels_
    label_count = defaultdict(int)
    for l in labels:
        label_count[l] += 1
    
    # filter RS, and get new X index 
    RS = set()
    X_idx_new = []
    for i in range(len(labels)):
        if label_count[labels[i]] == 1:
            RS.add(X_idx_pre[i])
        else:
            X_idx_new.append(X_idx_pre[i])
    
    return RS, X_idx_new

def ini_ds(model_ds, X_idx):
    ds_cluster_id = {i: [] for i in range(n_cluster)}
    for i in range(len(X_idx)):
        ds_cluster_id[kmeans_ds.labels_[i]].append(X_idx[i])

    ds_list = []  # store DS class
    for name, data_id in ds_cluster_id.items():
        # create and initialize DS class
        ds = DS_CS(name, data_id)
        ds_list.append(ds)
    return ds_list

def get_rs_cs(model_rs, X_idx):
    '''
    1. update rs
    2. initialize cs
    '''
    # count cluster
    labels = model_rs.labels_
    label_count = defaultdict(int)
    for l in labels:
        label_count[l] += 1
    
    # filter rs and cs cluster id
    rs_cluster_id = set()
    for key, val in label_count.items():
        if val == 1:
            rs_cluster_id.add(key)
        
    # filter RS, and store CS cluster
    RS = set()
    cs_cluster = defaultdict(list)
    for i in range(len(labels)):
        if labels[i] in rs_cluster_id:
            RS.add(X_idx[i])
        else:
            cs_cluster[labels[i]].append(X_idx[i])
    
    # initialize CS cluster
    cs_list = [] # store initicialized RS class
    for name, data_id in cs_cluster.items():
        cs = DS_CS(name, data_id)
        cs_list.append(cs)    
    
    return RS, cs_list       

def get_m_res(ds_list, cs_list, RS):
    n_discard = sum([ds.n for ds in ds_list])
    n_cs_cluster = len(cs_list)
    n_compression = sum([cs.n for cs in cs_list])
    n_retain = len(RS)
    res = [n_discard, n_cs_cluster, n_compression, n_retain]
    return res

def mahalanobis_dist(feature, ds_cs):
    dist_sq = 0
    centroid = ds_cs.get_centroid()
    var = ds_cs.cal_var()
    for i in range(len(feature)):
        dist_sq += ((feature[i] - centroid[i]) / var[i] ** 0.5) ** 2
    return dist_sq ** 0.5

def classify_data(feature, ds_list, cs_list):
    min_dist = float('inf')
    # 1. compare to ds
    for i in range(len(ds_list)):
        ds = ds_list[i]
        dist = mahalanobis_dist(feature, ds)
        if dist < min_dist and dist < alpha * d ** 0.5:
            min_dist = dist
            ds_position = i
    if min_dist != float('inf'):
        return (1, ds_position) 

    # 2. compare to cs
    for i in range(len(cs_list)):
        cs = cs_list[i]
        dist = mahalanobis_dist(feature, cs)
        if dist < min_dist and dist < alpha * d ** 0.5:
            min_dist = dist
            cs_position = i   
    if min_dist != float('inf'):
        return (0, cs_position) 
    # 3. assign to rs
    return (-1, -1)

def assign_ds_cs(candidate_ds_cs, ds_cs_list):
    for data_index, ds_cs_index in candidate_ds_cs:
        ds_cs = ds_cs_list[ds_cs_index]
        ds_cs.add_point(data_index)   
    return

def merge_CS(cs_list):
    while len(cs_list) > 1:
        dist_list = []
        for i in range(len(cs_list)):
            for j in range(i + 1, len(cs_list)):
                dist = mahalanobis_dist(cs_list[i].get_centroid(), cs_list[j])
                dist_list.append([i, j, dist])
        dist_list.sort(key = lambda x: x[2], reverse = True)
        
        i, j, min_dist = dist_list.pop()
        if min_dist >= 2 * d ** 0.5:
            break
            
        # merge cs2 to cs1
        cs1, cs2 = cs_list[i], cs_list[j]
        cs1.data_id |= cs2.data_id
        cs1.n += cs2.n
        for i in range(d):
            cs1.sum_feature[i] += cs2.sum_feature[i]
            cs1.sumsq_feature[i] += cs2.sumsq_feature[i]
        cs_list.pop(j)
        
    return cs_list

def merge_ds_cs(ds_list, cs_list):
    while cs_list:
        dist_list = [] # [cs_idx, ds_idx, dist]
        for i in range(len(cs_list)):
            for j in range(len(ds_list)):
                dist = mahalanobis_dist(cs_list[i].get_centroid(), ds_list[j])
                dist_list.append([i, j, dist])
        dist_list.sort(key = lambda x: x[2], reverse = True)
        
        cs_idx, ds_idx, min_dist = dist_list.pop()
        if min_dist >= 2 * d ** 0.5:
            break 
        # merge cs to ds
        ds, cs = ds_list[ds_idx], cs_list[cs_idx]
        ds.data_id |= cs.data_id
        ds.n += cs.n
        for i in range(d):
            ds.sum_feature[i] += cs.sum_feature[i]
            ds.sumsq_feature[i] += cs.sumsq_feature[i]
        cs_list.pop(cs_idx)    
    return ds_list

def get_final_res(ds_list):
    res = [-1] * len(data_map)
    for i in range(len(ds_list)):
        ds = ds_list[i]
        for idx in ds.data_id:
            res[idx] = i
    return res

if __name__ == '__main__':
    start = datetime.datetime.now()

    input_file_path = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_file_path = sys.argv[3]

    split_ratio = 0.2
    n_loop = int((1 - split_ratio) / 0.2)
    alpha = 2

    # build sc
    conf = SparkConf().setAppName("task1").setMaster("local[*]")
    sc = SparkContext(conf = conf)
    sc.setLogLevel("WARN")
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.driver.memory", "4g")

    data_rdd = sc.textFile(input_file_path).map(lambda x: x.split(','))\
                    .map(lambda x: [int(x[0]), [float(feature) for feature in x[2:]]])
    data_map = data_rdd.collectAsMap() # {data id(int): data features(list)}

    # get split proportion
    split = []
    for i in range(n_loop):
        if i == 0:
            split.append([split_ratio/(1-split_ratio), 1-split_ratio/(1-split_ratio)])
        elif i == n_loop - 1:
            split.append([1, 0])
        else:
            pre_split = split[-1][0]
            split.append([pre_split/(1-pre_split), 1-pre_split/(1-pre_split)])

    '''Step 1. Load 20% of the data randomly'''
    ini_data, rest_data0 = data_rdd.randomSplit([2, 8], 1)
    X_idx = ini_data.map(lambda x: x[0]).collect()
    X = ini_data.map(lambda x: x[1]).collect()

    d = len(X[0])

    '''Step 2. Run K-Means'''
    kmeans_rs1 = KMeans(n_clusters = 5 * n_cluster).fit(X)

    '''step 3. move all the clusters that contain only one point to RS'''
    RS, X2_idx = get_rs(kmeans_rs1, X_idx)

    '''Step 4. Run K-Means again to cluster the rest of the data points'''
    X2 = [data_map[i] for i in X2_idx]
    kmeans_ds = KMeans(n_clusters = n_cluster).fit(X2)

    '''Step 5. Generate the DS clusters (i.e.discard their points and generate statistics)'''
    ds_list = ini_ds(kmeans_ds, X2_idx)

    '''Step 6. Run K-Means on the points in the RS with a large K (5 times of the number of the input clusters) 
    to generate CS (clusters with more than one points) and RS (clusters with only one point)'''
    if len(RS) > 5 * n_cluster:
        # run k_means
        X3 = [data_map[i] for i in RS]
        kmeans_rs2 = KMeans(n_clusters = 5 * n_cluster).fit(X3)
        RS, cs_list = get_rs_cs(kmeans_rs2, X2_idx)
    else:
        # RS remains unchanged
        cs_list = []

    # store intermediate results
    m_res = {i: [] for i in range(5)}
    m_res[0] = get_m_res(ds_list, cs_list, RS)

    # loop start
    for i in range(n_loop):
        '''Step 7. Load another 20% of the data randomly'''
        exec('train_data, rest_data{} = rest_data{}.randomSplit(split[i], 1)'.format(i + 1, i))

        '''Step 8. For the new points, compare them to each of the DS using the Mahalanobis Distance and assign
        them to the nearest DS clusters if the distance is < 2√𝑑.'''
        # classify candidate to DS, CS, RS
        # x[1][0]: 1 -> DS, 0 -> CS, -1 -> RS
        data_assigned = train_data.map(lambda x: (x[0], classify_data(x[1], ds_list, cs_list)))
        # assign DS
        candidate_ds = data_assigned.filter(lambda x: x[1][0] == 1).map(lambda x: (x[0], x[1][1])).collect()
        assign_ds_cs(candidate_ds, ds_list)  
        '''Step 9. For the new points that are not assigned to DS clusters, using the Mahalanobis Distance and assign
        the points to the nearest CS clusters if the distance is < 2√𝑑'''
        # assign CS
        candidate_cs = data_assigned.filter(lambda x: x[1][0] == 0).map(lambda x: (x[0], x[1][1])).collect()
        assign_ds_cs(candidate_cs, cs_list)
        '''Step 10. For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS.'''
        # assign the rest to RS
        candidate_rs = data_assigned.filter(lambda x: x[1][0] == -1).map(lambda x: x[0]).collect()
        RS |= set(candidate_rs)  

        '''Step 11. Run K-Means on the RS with a large K (e.g., 5 times of the number of the input clusters) to
        generate CS (clusters with more than one points) and RS (clusters with only one point).'''
        if len(RS) > 5 * n_cluster:
            # run k_means
            X_idx = list(RS)
            X_new = [data_map[i] for i in X_idx]
            kmeans_rs = KMeans(n_clusters = 5 * n_cluster).fit(X_new)
            RS, cs_list = get_rs_cs(kmeans_rs, X_idx)

        '''Step 12. Merge CS clusters that have a Mahalanobis Distance < 2√𝑑.'''
        cs_list = merge_CS(cs_list)

        # store output
        m_res[i + 1] = get_m_res(ds_list, cs_list, RS)

    '''Last run: merge CS clusters with DS clusters that have a Mahalanobis Distance < 2√𝑑'''
    ds_list = merge_ds_cs(ds_list, cs_list)
    final_res = get_final_res(ds_list)

    # write results
    file = open(output_file_path, 'w')
    file.write('The intermediate results:\n')
    for key in sorted(m_res.keys()):
        file.write('Round {0}: {1}\n'.format(key + 1, str(m_res[key])[1:-1]))  
    file.write('\nThe clustering results:\n')
    for i in range(len(final_res)):
        file.write('{0},{1}\n'.format(i, final_res[i]))
    file.close()
    
    end = datetime.datetime.now()
    print((end - start).seconds)