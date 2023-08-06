from pyspark import SparkConf, SparkContext
import sys, datetime
from operator import add
from collections import defaultdict

# build sc
conf = SparkConf().setAppName("task1").setMaster("local[*]")
sc = SparkContext(conf = conf)
sc.setLogLevel("WARN")

def get_candidate(map_res, length):
    res = []
    
    for record in map_res:
        candidate = []
        
        if length == 1:
            # record: ('1', {'100', '101', '98', '102'})
            bkt = set()
            for item in record[1]:
                item2 = tuple([item])
                
                #item: '100'
                if item2 in candidate_all[length - 1]:
                    bkt.add(item2[0])
                    candidate.append(item2)
                   
            res.append((record[0], candidate, bkt))
            
        else:
            #record: 
            #('1',[['102', '98'],['101', '102'],['100', '102'],['100', '101'],['101', '98'],['100', '98']],
            #{'100', '101', '102', '98'})
            for item in record[1]:
                # item: ['102', '98']
                tuple_item = tuple(item)
                if tuple_item in candidate_all[length - 1]:
                    candidate.append(tuple_item)
            
            res.append((record[0], candidate, record[2]))     
            
    return res

def add_subset(pre_sub):
    res = []
    for idx, pres, bkt in pre_sub:
        
        # idx, pres, bkt:('1', [('100',), ('101',), ('98',), ('102',)], {'100', '101', '102', '98'}),
        # idx, pres, bkt:
        #('1',[('102', '98'),('101', '102'),('100', '102'),('100', '101'),('101', '98'),('100', '98')],{'100', '101', '102', '98'})
        subsets = []
        for pre in pres:
            for item in bkt:
                if item not in pre:
                    new = list(pre) + [item]
                    subsets.append(sorted(new[:]))

        subsets = [t for t in set(tuple(_) for _ in subsets)]
        
        res.append((idx, subsets, bkt))
    return res

def filter_candidate(map_res, support):
    count = defaultdict(int)
    for record in map_res:
        #record: 
        #('1',[('102', '98'),('101', '102'),('100', '102'),('100', '101'),('101', '98'),('100', '98')],{'100', '101', '102', '98'})
        for r in record[1]:
            count[r] += 1
            
    candidate = []
    for key in count:
        if count[key] >= support:
            candidate.append(key)
    
    return candidate

def a_prior(partitiondata):
    global candidate_all
    
    partition_bkt = list(partitiondata)
    support = s * (len(partition_bkt) / size)
    map_res = partition_bkt
    n = 1
    
    while True:
        pre_sub = get_candidate(map_res, n)
        map_res = add_subset(pre_sub)
        candidate = filter_candidate(map_res, support)
        if candidate == []:
            break
        candidate_all.append(candidate)
        n += 1
    
    return candidate_all


def stage_one(basket):
    global candidate_all
    
    single = basket.flatMap(lambda x: [[idx, 1] for idx in x[1]]).reduceByKey(add).filter(lambda x: x[1] >= s)\
            .map(lambda x: x[0]).collect()
    single_tuple = []
    for el in single:
        single_tuple.append(tuple([el]))
        
    candidate_all = [single_tuple]
    
    result = basket.mapPartitions(a_prior)\
                .flatMap(lambda x: x).distinct()\
                .map(lambda x: (len(x), x)).groupByKey().mapValues(list)\
                .sortBy(lambda x: x[0]).map(lambda x: x[1])
    
    return result.collect()


def count_candidate(partitiondata):
    data = list(partitiondata)
    count_res = defaultdict(int)
    
    for idx, bsk in data:
        # idx, bsk: ('1', {'100', '101', '102', '98'})
        for candidate in flat_poss_candidate:
            if set(candidate).issubset(bsk):
                count_res[candidate] += 1         

    return [(key,count) for key,count in count_res.items()]

def print_format(result_list):
    for i in range(len(result_list)):
        target = sorted(result_list[i])
        
        if i == 0:
            s = ''
            for element in target:
                new = "(\'" + str(element[0]) + "\'),"
                s += new
            s = s[:-1] + "\n\n"
        else:
            for element in target:
                new = str(element) + ','
                s += new
            s = s[:-1] + "\n\n"
    return s

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    
    case = sys.argv[1]
    s = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]
    candidate_all = []
    
    #read data into rdd
    smallRDD = sc.textFile(input_file_path).map(lambda x: x.split(','))
    first = smallRDD.take(1)

    # build basket
    if case == '1':
        basket = smallRDD.filter(lambda x: x != first[0]).groupByKey().mapValues(set)
    else:
        basket = smallRDD.filter(lambda x: x != first[0]).map(lambda x: [x[1], x[0]]).groupByKey().mapValues(set)
    
    size = basket.count()
    
    # Son stage 1
    poss_candidate = stage_one(basket)
    res_candidates = print_format(poss_candidate)
    
    # Son stage 2
    flat_poss_candidate = []
    for l in poss_candidate:
        flat_poss_candidate += l  
    
    true_candidate = basket.mapPartitions(count_candidate)\
                .reduceByKey(add).filter(lambda x: x[1] >= s).map(lambda x: x[0])\
                .map(lambda x: (len(x), x)).groupByKey().mapValues(list)\
                .sortBy(lambda x: x[0]).map(lambda x: x[1]).collect()
    
    res_frequent_items = print_format(true_candidate)
    
    
    # write output file
    with open(output_file_path, 'w+') as f:
        f.write('Candidates:\n' + res_candidates + 'Frequent Itemsets:\n' + res_frequent_items)
   
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).seconds
    print("Duration:",duration)


