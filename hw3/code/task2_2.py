from pyspark import SparkConf, SparkContext
import json, sys, datetime
from xgboost import XGBRegressor
from operator import add
import pandas as pd
import numpy as np

# build sc
conf = SparkConf().setAppName("task2.2").setMaster("local[*]")
sc = SparkContext(conf = conf)
sc.setLogLevel("WARN")
conf.set("spark.executor.memory", "4g")
conf.set("spark.driver.memory", "4g")

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
    #review_path = folder_file + 'review_train.json'

    trainRDD = sc.textFile(train_file).map(lambda x: x.split(','))
    firstline = trainRDD.first()
    train_set = trainRDD.filter(lambda x: x != firstline)

    testRDD = sc.textFile(test_file).map(lambda x: x.split(','))
    first = testRDD.first()
    test_set = testRDD.filter(lambda x: x != first)


    # extract features

    # 188593
    # (business, stars, review count)
    businessRDD = sc.textFile(business_data_path).map(lambda x: json.loads(x))\
                        .map(lambda x: (x['business_id'], (x['stars'], x['review_count'])))
    # 157075
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
    
    # 17538
    # ((user, business), like number)
    tipRDD = sc.textFile(tip_path).map(lambda x: json.loads(x))\
                        .map(lambda x: ((x['user_id'], x['business_id']), x['likes']))
    tip_feature = tipRDD.collectAsMap()
    # 1518169
    # (user_id, number of reviews, average star, elite years)
    userRDD = sc.textFile(user_data_path).map(lambda x: json.loads(x))
    user_feature = userRDD.map(lambda x: (x['user_id'], (x['review_count'], x['average_stars'],\
                                 int(x['yelping_since'][:4]), x['fans'], x['useful'], x['useful']+x['funny']+x['cool'],\
                                 0 if x["elite"] == 'None' else (len(x["elite"])+2) // 6,\
                                 x['compliment_hot']+x['compliment_more']+x['compliment_plain']+x['compliment_cool']\
                                                         +x['compliment_funny']+x['compliment_writer'])))
    user_feature_map = user_feature.collectAsMap()


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

    # train model
    model = XGBRegressor(n_estimators = 500, learning_rate = 0.1, max_depth = 5, eval_metric='rmse')\
                .fit(X_train, y_train)

    # test dataset
    test_rdd = test_set.map(lambda x: ((x[0], x[1]), user_feature_map[x[0]], bsn_feature_map[x[1]], \
                                     tip_feature[(x[0], x[1])] if (x[0], x[1]) in tip_feature else 0, \
                                      photo_feature[x[1]] if x[1] in photo_feature else (0, 0)))\
                        .map(lambda x: (x[0], x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], x[1][5], x[1][6], x[1][7]\
                                    , x[2][0], x[2][1], x[2][2], x[3], x[4][0], x[4][1]))

    test = test_rdd.collect()
    test_df = pd.DataFrame(test, columns = ['user_bsn','u_reviews', 'u_average_star', 'u_since', 'u_fans', 'u_useful', 'u_good',\
                                    'u_elite_years','u_compliment','b_stars','b_review_count', 'b_check_in_times', 'tip', \
                                    'photo_food', 'photo_env']).set_index('user_bsn')

    user_bsn_id = test_df.index
    y_predict = model.predict(test_df)

    # write into file
    file = open(output_file_path, 'w')
    file.write('business_id_1, business_id_2, prediction')
    for i in range(len(y_predict)):
        file.write('\n')
        file.write(str(user_bsn_id[i][0]) + ',' + str(user_bsn_id[i][1] + ',' + str(y_predict[i])))
    file.close()

    end_time = datetime.datetime.now()
    print((end_time - start_time).seconds)