from pyspark import SparkConf, SparkContext
import sys, datetime, random
from operator import add
from collections import defaultdict


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    # build sc
    conf = SparkConf().setAppName("task1").setMaster("local[*]")
    sc = SparkContext(conf = conf)
    sc.setLogLevel("WARN")
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.driver.memory", "4g")

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    n = 100
    a = [143, 257, 193, 103, 27, 117, 239, 197, 13, 229, 117, 109, 221, 149, 113, 137, 277, 269, 163, 157, 221, 131, 19, 193, 89, 73, 213, 41, 281, 79, 167, 249, 99, 163, 279, 27, 39, 263, 121, 51, 209, 249, 101, 243, 151, 99, 79, 87, 157, 153, 37, 249, 209, 3, 241, 293, 143, 57, 269, 89, 109, 249, 211, 101, 209, 31, 87, 127, 221, 11, 81, 257, 293, 41, 131, 157, 269, 143, 83, 127, 111, 277, 201, 169, 163, 117, 199, 181, 57, 111, 59, 71, 51, 179, 241, 271, 11, 9, 41, 257]
    b = [730, 593, 220, 736, 87, 551, 415, 262, 980, 650, 869, 588, 184, 51, 779, 514, 954, 458, 854, 49, 816, 487, 233, 80, 23, 412, 776, 938, 396, 567, 310, 874, 440, 57, 70, 176, 193, 979, 93, 511, 475, 747, 34, 690, 983, 649, 13, 305, 464, 861, 341, 926, 12, 27, 425, 48, 924, 703, 226, 796, 734, 836, 560, 203, 194, 894, 695, 498, 410, 859, 9, 243, 90, 613, 595, 667, 959, 341, 954, 547, 190, 316, 947, 524, 51, 341, 375, 376, 945, 595, 106, 822, 205, 271, 125, 667, 138, 255, 486, 224]

    n_band = 50
    r = 2
    threshold = 0.5


    # read data
    yelpRDD = sc.textFile(input_file_path).map(lambda x: x.split(','))
    firstline = yelpRDD.first()
    yelpRDD = yelpRDD.filter(lambda x: x != firstline)

    user_id = yelpRDD.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()
    n_user = len(user_id)
    business_id = yelpRDD.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
    n_business = len(business_id)
    
    final_matrix = yelpRDD.map(lambda x: (business_id[x[1]], user_id[x[0]]))\
                        .groupByKey().mapValues(set).collectAsMap()
    
    
    id_business = {}
    for key, val in business_id.items():
        id_business[val] = key
    

    input_matrix = yelpRDD.map(lambda x: [user_id[x[0]], business_id[x[1]]])\
                        .groupByKey().mapValues(set)

    def hash_idx(record):
        record = [record[0], list(record[1])]
        idx = record[0]
        hash_res = []
        for i in range(n):
            new_idx = (a[i] * idx + b[i]) % n_user
            hash_res.append(new_idx)
        record[0] = hash_res

        return record  


    # min-hash part
    # key: business_id, val: [user_ids]
    signatures = input_matrix.map(hash_idx)\
                        .map(lambda x: ((idx,x[0]) for idx in x[1])).flatMap(lambda x: x)\
                        .groupByKey().mapValues(list)\
                        .map(lambda x: (x[0], [min(idx_l) for idx_l in zip(*x[1])]))

    # signature_matrix = signatures.collectAsMap()

    def split_into_band(sig):
        band_data = sig[1]
        new_band_data = []
        for i in range(0, len(band_data), r):
            new_band_data.append(band_data[i: i + r])
        return (sig[0], new_band_data)

    sig_band_res = signatures.map(split_into_band).collect()


    # find candidate pairs
    candidate_pair = set()
    for i in range(n_band):
        bucket = defaultdict(set)
        for j in range(len(sig_band_res)):
            business_record = sig_band_res[j]
            idx = business_record[1][i]
            bucket[tuple(idx)].add(business_record[0])
        for key, val in bucket.items():
            length = len(val)
            val = list(val)
            if length > 1:
                for p in range(length):
                    for q in range(p + 1, length):
                        pair = tuple(sorted([val[p], val[q]]))
                        if pair not in candidate_pair:
                            candidate_pair.add(pair)



    def find_similar(pair):
        sig1 = set(final_matrix[pair[0]])
        sig2 = set(final_matrix[pair[1]])
        inter = len(sig1.intersection(sig2))
        union = len(sig1) + len(sig2) - inter
        return inter / union

    # filter similar pairs
    candidateRDD = sc.parallelize(candidate_pair)
    similar_pair = candidateRDD.map(lambda x: (x, find_similar(x)))\
                        .filter(lambda x: x[1] >= threshold)\
                        .map(lambda x: (tuple(sorted([id_business[x[0][0]], id_business[x[0][1]]])), x[1]))\
                        .sortBy(lambda x: (x[0][0], x[0][1]))\
                        .collect()

    # write into file
    file = open(output_file_path, 'w')
    file.write('business_id_1, business_id_2, similarity')
    for pair, sim in similar_pair:
        file.write('\n')
        file.write(pair[0] + ',' + pair[1] + ',' + str(sim))
    file.close()

    end_time = datetime.datetime.now()
    print((end_time - start_time).seconds)