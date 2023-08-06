from pyspark import SparkConf, SparkContext
import json, sys, datetime
from xgboost import XGBRegressor
from operator import add
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from collections import defaultdict
import math

conf = SparkConf().setAppName("task2.3").setMaster("local[*]")
sc = SparkContext(conf = conf)
sc.setLogLevel("WARN")
conf.set("spark.executor.memory", "4g")
conf.set("spark.driver.memory", "4g")

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
      
    return weight, count1


def predict_rating(business, user, weight):
    length = len(weight)
    if length <= 7 or count1 == 0:
        return -1
    
    base = res_avg_rating[business]
    rating = 0
    weight_sum = 0
    for pair, w in weight:
        user_id = pair[0]
        bsn_id = pair[1]
        weight_sum += w
        r = bsn_user[pair[1]][user]
        rating += r * w
    res = rating / weight_sum
    
    return 0.4 * res + 0.6 * base 

def item_cf(user, business):
    global count1
    if user not in user_bsn or business not in bsn_user:
        return -1
    
    candidate_pair = get_candidate(user, business)
    weight, count1 = get_weight(business, candidate_pair)
    rating = predict_rating(business, user, weight)
    return rating

def item_cf_predict(train_set, test_set):
    global bsn_user, user_bsn, avg_rating, res_avg_rating
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
#     # {user id: avg rating}
#     avg_user_rating = train_set.map(lambda x: (x[0],(x[1], float(x[2]))))\
#                         .groupByKey().mapValues(dict).map(lambda x: (x[0], cal_avg(x))).collectAsMap()

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
    result = predict_ratings.map(lambda x: (tuple(x[0]), x[1])).collectAsMap()
    return result


def model_cf_predict(train_set, test_set):
    # 1. extract features
    # (business, stars, review count)
    businessRDD = sc.textFile(business_data_path).map(lambda x: json.loads(x))\
                        .map(lambda x: (x['business_id'], (x['stars'], x['review_count'])))
    # (business, check in times)
    checkinRDD = sc.textFile(checkin_path).map(lambda x: json.loads(x))\
                        .map(lambda x: (x['business_id'], len('time')))
    # (business, stars, review count, check in times)
    business_feature = businessRDD.leftOuterJoin(checkinRDD)\
                                .map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][1] if x[1][1] != None else 0)))
    bsn_feature_map = business_feature.collectAsMap()
    # photo (food+drink+menu, inside+outside)
    def encode(record):
        food = 0
        env = 0
        for tag in record[1]:
            if tag in ['food','drink', 'menu']:
                food += 1
            if tag in ['inside','outside']:
                env += 1
        return (record[0], (food, env))
    photoRDD = sc.textFile(photo_path).map(lambda x: json.loads(x))\
                        .map(lambda x: ((x['business_id']), x['label'])).groupByKey().mapValues(list)\
                        .map(lambda x: encode(x))
    photo_feature = photoRDD.collectAsMap()
    # ((user, business), like number)
    tipRDD = sc.textFile(tip_path).map(lambda x: json.loads(x))\
                        .map(lambda x: ((x['user_id'], x['business_id']), x['likes']))
    tip_feature = tipRDD.collectAsMap()
    # (user_id, number of reviews, average star, elite years)
    userRDD = sc.textFile(user_data_path).map(lambda x: json.loads(x))
    user_feature = userRDD.map(lambda x: (x['user_id'], (x['review_count'], x['average_stars'],\
                                 int(x['yelping_since'][:4]), x['fans'], x['useful'], x['useful']+x['funny']+x['cool'],\
                                 0 if x["elite"] == 'None' else (len(x["elite"])+2) // 6,\
                                 x['compliment_hot']+x['compliment_more']+x['compliment_plain']+x['compliment_cool']\
                                                         +x['compliment_funny']+x['compliment_writer'])))
    user_feature_map = user_feature.collectAsMap()
    
    # 2. get training dataset
    # ((user, business), u_reviews, u_average_star, u_since, u_fans, u_useful, u_good, u_elite_years, u_compliment
    # b_stars, b_review_count, b_check_in_times, tip, photo_food, photo_env, rating)
    train_rdd = train_set.map(lambda x: ((x[0], x[1]), user_feature_map[x[0]], bsn_feature_map[x[1]], \
                                      tip_feature[(x[0], x[1])] if (x[0], x[1]) in tip_feature else 0, \
                                      photo_feature[x[1]] if x[1] in photo_feature else (0, 0), float(x[2])))\
                        .map(lambda x: (x[0], x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], x[1][5], x[1][6], x[1][7]\
                                    , x[2][0], x[2][1], x[2][2], x[3], x[4][0], x[4][1], x[5]))
    train = train_rdd.collect()

    train_df = pd.DataFrame(train, columns = ['user_bsn','u_reviews', 'u_average_star', 'u_since', 'u_fans', 'u_useful', 'u_good',\
                                    'u_elite_years','u_compliment','b_stars','b_review_count', 'b_check_in_times', 'tip', \
                                    'photo_food', 'photo_env', 'rating']).set_index('user_bsn')
    X_train = train_df.iloc[:,:-1]
    y_train = train_df.iloc[:,-1]

    # 3. train model
    model = XGBRegressor(n_estimators = 500, learning_rate = 0.1, max_depth = 5, eval_metric='rmse')\
                .fit(X_train, y_train)

    # 4. get test dataset
    test_rdd = test_set.map(lambda x: ((x[0], x[1]), user_feature_map[x[0]], bsn_feature_map[x[1]], \
                                     tip_feature[(x[0], x[1])] if (x[0], x[1]) in tip_feature else 0, \
                                      photo_feature[x[1]] if x[1] in photo_feature else (0, 0)))\
                        .map(lambda x: (x[0], x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], x[1][5], x[1][6], x[1][7]\
                                    , x[2][0], x[2][1], x[2][2], x[3], x[4][0], x[4][1]))

    test = test_rdd.collect()
    test_df = pd.DataFrame(test, columns = ['user_bsn','u_reviews', 'u_average_star', 'u_since', 'u_fans', 'u_useful', 'u_good',\
                                    'u_elite_years','u_compliment','b_stars','b_review_count', 'b_check_in_times', 'tip', \
                                    'photo_food', 'photo_env']).set_index('user_bsn')
    
    # 5. predict
    user_bsn_id = test_df.index
    y_predict = model.predict(test_df)
    
    return user_bsn_id, y_predict

if __name__ == '__main__':
    start_time = datetime.datetime.now()

    folder_file = sys.argv[1]
    train_file = folder_file + 'yelp_train.csv'
    test_file = sys.argv[2]
    output_file_path = sys.argv[3]
    business_data_path = folder_file + 'business.json'
    checkin_path = folder_file + 'checkin.json'
    tip_path = folder_file + 'tip.json'
    user_data_path = folder_file + 'user.json'
    photo_path = folder_file + 'photo.json'

    trainRDD = sc.textFile(train_file).map(lambda x: x.split(','))
    firstline = trainRDD.first()
    train_set = trainRDD.filter(lambda x: x != firstline)

    testRDD = sc.textFile(test_file).map(lambda x: x.split(','))
    first = testRDD.first()
    test_set = testRDD.filter(lambda x: x != first)
    
    # item_based cf
    res_item_dic = item_cf_predict(train_set, test_set)
    
    # model_based cf
    idx_pair, res_model_array = model_cf_predict(train_set, test_set)
    
    # write into file
    file = open(output_file_path, 'w')
    file.write('business_id_1, business_id_2, prediction')
    for i in range(len(res_model_array)):
        file.write('\n')
        key = idx_pair[i]
        if res_item_dic[key] == -1:
            file.write(str(key[0]) + ',' + str(key[1]) + ',' + str(res_model_array[i]))
        else:
            value = 0.2 * res_item_dic[key]+ 0.8 * res_model_array[i]
            file.write(str(key[0]) + ',' + str(key[1]) + ',' + str(value))
    file.close()

    end_time = datetime.datetime.now()
    print((end_time - start_time).seconds)    