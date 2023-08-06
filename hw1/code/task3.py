from pyspark import SparkConf, SparkContext
import json, sys, operator, datetime

if __name__ == '__main__':
    
    review_path = sys.argv[1]
    business_path = sys.argv[2]
    output_path_a = sys.argv[3]
    output_path_b = sys.argv[4]

    def m1_spark():

        # build sc
        conf = SparkConf().setAppName("task3").setMaster("local[*]")
        sc = SparkContext(conf = conf)
        sc.setLogLevel('WARN')

        # read data into rdd
        reviewRDD = sc.textFile(review_path).map(lambda record: json.loads(record))\
                .map(lambda record: (record['business_id'], record['stars']))

        businessRDD = sc.textFile(business_path).map(lambda record: json.loads(record))\
                .map(lambda record: (record['business_id'], record['city']))


        top10_city = reviewRDD\
                .leftOuterJoin(businessRDD).filter(lambda x: x[1][1] != None)\
                .map(lambda x: (x[1][1], x[1][0]))\
                .groupByKey().mapValues(list)\
                .map(lambda x: (x[0], sum(x[1]) / len(x[1])))\
                .sortBy(lambda x: (-x[1], x[0]))\
                .map(lambda x: round(x[1], 1))\
                .take(10)

        return top10_city

    def m2_python():

        # read review data
        review_data = []
        with open(review_path, encoding='utf-8') as review:
            for record in review:
                data_json = json.loads(record)
                review_data.append([data_json['business_id'],data_json['stars']])

        # read business data
        business_data = {}
        with open(business_path, encoding='utf-8') as business:
            for record in business:
                data_json = json.loads(record)
                business_data[data_json['business_id']] = data_json['city']

        # star: {city: [sum_of_stars, num_of_record]}
        star = {}
        for pair in review_data:
            if pair[0] in business_data:
                city = business_data[pair[0]]
                if city in star:
                    star[city][0] += pair[1]
                    star[city][1] += 1
                else:
                    star[city] = [pair[1], 1]

        # result: [city, avg_stars]
        result = []
        for city in star:
            avg_star = star[city][0] / star[city][1]
            result.append([city, avg_star])


        #get top10 cities
        result = sorted(result, key = lambda x: (-x[1], x[0]))
        top10_city = result[:10]
        for i in range(10):
            top10_city[i][1] = round(top10_city[i][1], 1)


        return top10_city


    def print_result(top10_city):
        print("city,stars")
        for pair in top10_city:
            print(str(pair[0])+","+str(pair[1]))



    # run the code here

    # using spark
    start = datetime.datetime.now()
    top10_city = m1_spark()
    end = datetime.datetime.now()
    m1 = (end - start).seconds
    print_result(top10_city)

    # output file for Question A
    file = open(output_path_a, 'w')
    file.write('city,stars')
    for pair in top10_city:
        file.write('\n')
        file.write(str(pair[0])+","+str(pair[1]))
    file.close()



    # not using spark
    start = datetime.datetime.now()
    top10_city = m2_python()
    end = datetime.datetime.now()
    m2 = (end - start).seconds
    print_result(top10_city)


    output_b = {}
    output_b['m1'] = m1
    output_b['m2'] = m2
    output_b['reason'] =  'If the dataset is so large, m1 method is faster than m2 method. There are many reasons. \
    First, Spark is designed in a way that it transforms data in-memory and not in disk I/O. \
    Hence, it cut off the processing time of read/write cycle to disk and storing intermediate data in-memory.\
    Second, Spark uses RDD data structure, which can be partitioned and computed on different nodes of a cluster.'


    # output file for Question B
    with open(output_path_b, 'w') as f:
        json.dump(output_b, f)