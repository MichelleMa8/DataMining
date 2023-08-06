from pyspark import SparkConf, SparkContext
import json, sys


if __name__ == '__main__':

    # create sc
    conf = SparkConf().setAppName("task1").setMaster("local[*]")
    sc = SparkContext(conf = conf)
    sc.setLogLevel('WARN')

    #read data into rdd
    input_file_path = "C:/Users/surface/Desktop/553/hw1/data/yelp_academic_dataset_review.json"
    reviewRDD = sc.textFile(input_file_path)

    #n_review
    review = reviewRDD.map(lambda record: json.loads(record))
    n_review = review.count()

    #n_review_2018
    review_18 = review.filter(lambda record: int(record['date'][:4]) == 2018)
    n_review_2018 = review_18.count()

    #n_user
    user = review.map(lambda record: record['user_id']).distinct()
    n_user = user.count()

    #top10_user
    usr_review = review.map(lambda record: [record['user_id'], record['review_id']]).groupByKey().mapValues(len)\
                    .sortBy(lambda x: (-x[1], x[0])).take(10)
    usr_list = []
    for record in usr_review:
        usr_list.append([record[0], record[1]])

    #n_business
    bsn = review.map(lambda record: record['business_id']).distinct()
    n_business = bsn.count()

    #top10_business
    bsn_review = review.map(lambda record: [record['business_id'], record['review_id']]).groupByKey().mapValues(len)\
                    .sortBy(lambda x:(-x[1], x[0])).take(10)
    bsn_list = []
    for record in bsn_review:
        bsn_list.append([record[0], record[1]])

    #output result in dict
    result = {}
    result['n_review'] = n_review
    result['n_review_2018'] = n_review_2018
    result['n_user'] = n_user
    result['top10_user'] = usr_list
    result['n_business'] = n_business
    result['top10_business'] = bsn_list

    #output result in json
    output_file_path = "C:/Users/surface/Desktop/output2.json"
    with open(output_file_path, 'w') as f:
        json.dump(result, f)