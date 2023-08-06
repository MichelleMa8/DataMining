from pyspark import SparkConf, SparkContext
import sys, datetime
from operator import add
from collections import defaultdict
import math

def get_candidate(user, business):
    pre_candidate = user_bsn[user]
    candidate_user = set(bsn_user[business].keys())
    candidate = set()
    for u in candidate_user:
        candidate |= user_bsn[u]
    candidate.discard(business)
    return candidate & pre_candidate

def pearson_sim(b1_target, b2):  
    users = set(bsn_user[b1_target].keys()) & set(bsn_user[b2].keys())        
    
    numerator = 0
    denominator1 = 0
    denominator2 = 0 
    
    avg_r1 = avg_rating[b1_target]
    avg_r2 = avg_rating[b2]
    
    for u in users:
        r1 = bsn_user[b1_target][u] - avg_r1
        r2 = bsn_user[b2][u] - avg_r2
        numerator += r1 * r2
        denominator1 += r1 ** 2
        denominator2 += r2 ** 2
           
    # if denominator = 0, return 0
    if denominator1 == 0 or denominator2 == 0:
        return 0
    
    similarity = numerator / (math.sqrt(denominator1 * denominator2))
    return max(0, similarity)

def get_weight(business, candidate_pair):
    weight = []
    for b2 in candidate_pair:
        similarity= pearson_sim(business, b2)
        if similarity != 0:
            weight.append([(business, b2), similarity])
    
    count1 = 0
    count2 = 0
    for i in range(len(weight)):
        pair, w = weight[i]
        if w >= 0.5:
            count1 += 1
        elif w >= 0.3:
            count2 += 1            
      
    return weight, count1, count2

def predict_rating(business, user, weight):
    length = len(weight)
    base = res_avg_rating[business]
    if length == 0 or count1 + count2 == 0:
        return base

    rating = 0
    weight_sum = 0
    for pair, w in weight:
        user_id = pair[0]
        bsn_id = pair[1]
        if user_id in user_bsn:
            f_bsn = 1 / (1 + math.log(len(user_bsn[user_id])))
        else:
            f_bsn = 1

        if bsn_id in bsn_user:
            f_user = 1 / (1 + math.log(len(bsn_user[bsn_id])))
        else:
            f_user = 1
        weight_sum += w * f_user * f_bsn
        r = bsn_user[pair[1]][user]
        rating += r * w * f_user * f_bsn
    res = rating / weight_sum
    
    if count1 == 0 and count2 != 0:
        return 0.2 * res + 0.8 * base
    else:
        return 0.4 * res + 0.6 * base  


def item_cf(user, business):
    global count1, count2
    if user not in user_bsn:
        if business not in bsn_user:
            return 3
        return res_avg_rating[business]
    if business not in bsn_user:
        return avg_user_rating[user]
    
    candidate_pair = get_candidate(user, business)
    weight, count1, count2 = get_weight(business, candidate_pair)
    rating = predict_rating(business, user, weight)
    return rating
    

if __name__ == '__main__':
    start_time = datetime.datetime.now()

    # build sc
    conf = SparkConf().setAppName("task2.1").setMaster("local[*]")
    sc = SparkContext(conf = conf)
    sc.setLogLevel("WARN")
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.driver.memory", "4g")
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file_path = sys.argv[3]

    # read data
    trainRDD = sc.textFile(train_file).map(lambda x: x.split(','))
    firstline = trainRDD.first() #['user_id', 'business_id', 'stars']
    train_set = trainRDD.filter(lambda x: x != firstline)

    testRDD = sc.textFile(test_file).map(lambda x: x.split(','))
    first = testRDD.first()
    test_set = testRDD.filter(lambda x: x != first).map(lambda x: (x[0], x[1]))  # user, business

    # (business, {user: rate, ...})
    bsn_user_rdd = train_set.map(lambda x: (x[1],(x[0], float(x[2]))))\
                        .groupByKey().mapValues(dict)
    bsn_user = bsn_user_rdd.collectAsMap()

    # (user, set(business, ...))
    user_bsn = train_set.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()

    def cal_avg(record):
        ratings = record[1].values()
        avg = round(sum(ratings) / len(ratings), 4)
        return avg 

    # {business id: avg rating}
    avg_rating = bsn_user_rdd.map(lambda x: (x[0], cal_avg(x))).collectAsMap()
    # {user id: avg rating}
    avg_user_rating = train_set.map(lambda x: (x[0],(x[1], float(x[2]))))\
                        .groupByKey().mapValues(dict).map(lambda x: (x[0], cal_avg(x))).collectAsMap()
    
    def weight_avg(record):
        ratings = record[1].values()
        length = len(ratings)
        if length < 3:
            avg = sum(ratings) / len(ratings)
            avg -= (avg - 3) * 0.5
        elif length < 8:
            avg = sum(ratings) / len(ratings)
        else:
            avg = (sum(ratings) - max(ratings) - min(ratings)) / (len(ratings) - 2)
        return avg
    
    res_avg_rating = bsn_user_rdd.map(lambda x: (x[0], weight_avg(x))).collectAsMap()

    predict_ratings = test_set.map(lambda x: (x, item_cf(x[0], x[1])))
    result = predict_ratings.collect()

    # write into file
    file = open(output_file_path, 'w')
    file.write('user_id, business_id, prediction')
    for pair, rating in result:
        file.write('\n')
        file.write(pair[0] + ',' + pair[1] + ',' + str(rating))
    file.close()
    
    end_time = datetime.datetime.now()
    print((end_time - start_time).seconds)
