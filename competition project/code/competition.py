from pyspark import SparkConf, SparkContext
import json, sys, datetime
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from operator import add

def extract_features():
    # business_id: (stars, review count)   188593   all
    bsn_feature = sc.textFile(business_data_path).map(lambda x: json.loads(x))\
                        .map(lambda x: (x['business_id'], (x['stars'], x['review_count']))).collectAsMap()
    # business_id: (checkin days, total checkin days)   157075   lack 532
    checkin_feature = sc.textFile(checkin_path).map(lambda x: json.loads(x))\
                        .map(lambda x: (x['business_id'], (len(x['time']), sum(x['time'].values()))))\
                        .collectAsMap()
    # business_id: (food+drink+menu, inside+outside) 32976        lack 128374
    def encode(record):
        food = 0
        env = 0
        for tag in record[1]:
            if tag in ['food','drink', 'menu']:
                food += 1
            if tag in ['inside','outside']:
                env += 1
        return (record[0], (food, env))
    photo_feature = sc.textFile(photo_path).map(lambda x: json.loads(x))\
                        .map(lambda x: ((x['business_id']), x['label'])).groupByKey().mapValues(list)\
                        .map(lambda x: encode(x)).collectAsMap()
    # (user_id, business_id): tip times     999945       lack 409824
    tip_feature = sc.textFile(tip_path).map(lambda x: json.loads(x))\
                        .map(lambda x: ((x['user_id'], x['business_id']), 1))\
                        .reduceByKey(add).collectAsMap()
    # user_id: (number of reviews, average star, yelp since, fans, votes sent by users, 
    #           elite years, nearest elite  year, compliments received by users)
    # 1518169       all
    userRDD = sc.textFile(user_data_path).map(lambda x: json.loads(x))
    user_feature = userRDD.map(lambda x: (x['user_id'],\
                                            (x['review_count'], x['average_stars'],\
                                            2019 - int(x['yelping_since'][:4]), x['fans'], x['useful']+x['funny']+x['cool'],\
                                            0 if x["elite"] == 'None' else (len(x["elite"])+2) // 6,\
                                            0 if x["elite"] == 'None' else int(x["elite"][-4:]) - 2004,\
                                            x['compliment_hot']+x['compliment_more']+x['compliment_plain']\
                                            +x['compliment_cool']+x['compliment_funny']+x['compliment_writer'])))\
                            .map(lambda x: (x[0], (x[1][0],x[1][1],x[1][2],x[1][3],x[1][4],x[1][5],x[1][6],\
                                                   0 if x[1][0] == 0 else x[1][7] / x[1][0])))\
                            .collectAsMap()
    return bsn_feature, checkin_feature, photo_feature, tip_feature, user_feature

def train_xgb(X_train, y_train):
    model = XGBRegressor(n_estimators = 500, learning_rate = 0.1, max_depth = 5, eval_metric='rmse')\
                .fit(X_train, y_train)
    return model

if __name__ == '__main__':
    start_time = datetime.datetime.now()

    conf = SparkConf().setAppName("competition").setMaster("local[*]")
    sc = SparkContext(conf = conf)
    sc.setLogLevel("WARN")
    conf.set("spark.executor.memory", "32g")
    conf.set("spark.driver.memory", "32g")

    folder_file = sys.argv[1]
    train_file = folder_file + 'yelp_train.csv'
    test_file = sys.argv[2]
    output_file_path = sys.argv[3]
    business_data_path = folder_file + 'business.json'
    checkin_path = folder_file + 'checkin.json'
    tip_path = folder_file + 'tip.json'
    user_data_path = folder_file + 'user.json'
    photo_path = folder_file + 'photo.json'
    xgb_model_file = './xgb.pickle'

    trainRDD = sc.textFile(train_file).map(lambda x: x.split(','))
    firstline = trainRDD.first()  # user, bsn, stars
    train_set = trainRDD.filter(lambda x: x != firstline).map(lambda x: (x[0], x[1], float(x[2])))

    # 142044
    testRDD = sc.textFile(test_file).map(lambda x: x.split(','))
    first = testRDD.first()
    test_set = testRDD.filter(lambda x: x != first).map(lambda x: (x[0], x[1], float(x[2])))

    bsn_feature, checkin_feature, photo_feature, tip_feature, user_feature = extract_features()

    train_rdd = train_set.map(lambda x: ((x[0], x[1]),user_feature[x[0]],bsn_feature[x[1]],\
                                        checkin_feature[x[1]] if x[1] in checkin_feature else (0, 0),\
                                        photo_feature[x[1]] if x[1] in photo_feature else (0, 0),\
                                        tip_feature[(x[0], x[1])] if (x[0], x[1]) in tip_feature else 0,\
                                        x[2]))\
                    .map(lambda x: (x[0], x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], x[1][5], x[1][6], x[1][7],\
                                x[2][0], x[2][1], x[3][0], x[3][1], x[4][0], x[4][1], x[5], x[6]))
    train_data = train_rdd.collect()
    train_df = pd.DataFrame(train_data, 
                        columns = 
                        ['user_bsn','u_reviews', 'u_star', 'u_years', 'u_fans', 'u_votes','u_elite','u_elite_1','u_compliment',\
                        'b_stars','b_reviews', 'b_checkin','b_checkin_total','b_photo_food', 'b_photo_env', 'tip','rating'])\
                .set_index('user_bsn')
    X_train = train_df.iloc[:,:-1]
    y_train = train_df.iloc[:,-1]

    test_rdd = test_set.map(lambda x: ((x[0], x[1]),user_feature[x[0]],bsn_feature[x[1]],\
                                        checkin_feature[x[1]] if x[1] in checkin_feature else (0, 0),\
                                        photo_feature[x[1]] if x[1] in photo_feature else (0, 0),\
                                        tip_feature[(x[0], x[1])] if (x[0], x[1]) in tip_feature else 0,\
                                        x[2]))\
                    .map(lambda x: (x[0], x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], x[1][5], x[1][6], x[1][7],\
                                x[2][0], x[2][1], x[3][0], x[3][1], x[4][0], x[4][1], x[5], x[6]))
    test_data = test_rdd.collect()
    test_df = pd.DataFrame(test_data, 
                        columns = 
                        ['user_bsn','u_reviews', 'u_star', 'u_years', 'u_fans', 'u_votes','u_elite','u_elite_1','u_compliment',\
                        'b_stars','b_reviews', 'b_checkin','b_checkin_total','b_photo_food', 'b_photo_env', 'tip', 'rating'])\
                .set_index('user_bsn')
    X_test = test_df.iloc[:,:-1]
    y_test = test_df.iloc[:,-1]

    # train model
    xgb = train_xgb(X_train, y_train)
    # # load model
    # with open(xgb_model_file, 'rb') as file:
    #     xgb = pickle.load(file)

    user_bsn_id = test_df.index
    xgb_predict = xgb.predict(X_test)

    # write into file
    file = open(output_file_path, 'w')
    file.write('user_id, business_id, prediction')
    for i in range(len(xgb_predict)):
        file.write('\n')
        key = user_bsn_id[i]
        file.write(str(key[0]) + ',' + str(key[1]) + ',' + str(xgb_predict[i]))
    file.close()

    end_time = datetime.datetime.now()
    print((end_time - start_time).seconds)