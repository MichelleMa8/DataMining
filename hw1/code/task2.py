from pyspark import SparkConf, SparkContext
import json, sys, datetime

if __name__ == '__main__':

    # build sc
    conf = SparkConf().setAppName("task2").setMaster("local[*]")
    sc = SparkContext(conf = conf)
    sc.setLogLevel("WARN")


    #read data into rdd as json format
    input_file_path = sys.argv[1]
    reviewRDD = sc.textFile(input_file_path).map(lambda record: json.loads(record))


    # function to get top10 business
    def get_top_business(rdd_data, dft, partition):
        review = rdd_data.map(lambda record: [record["business_id"], record["review_id"]])

        # record default n_partition
        if dft == True:
            n_partition = review.getNumPartitions()


        # customized n_partition
        else:
            review = review.partitionBy(partition)
            n_partition = partition

        n_item = review.glom().map(len).collect()

        #get top10 business
        business = review\
            .groupByKey().mapValues(len)\
            .sortBy(lambda x: (-x[1], x[0]))\
            .take(10)


        return n_item, n_partition 


    #result part


    # default
    start = datetime.datetime.now()
    n_item, n_partition = get_top_business(reviewRDD, True, 0)
    end = datetime.datetime.now()
    exe_time = (end - start).seconds

    # result for default partition
    default = {}
    default["n_partition"] = n_partition
    default["n_items"] = n_item
    default["exe_time"] = exe_time


    # customized
    partition_ = int(sys.argv[3])
    start = datetime.datetime.now()
    n_item, n_partition = get_top_business(reviewRDD, False, partition_)
    end = datetime.datetime.now()
    exe_time = (end - start).seconds

    # result for customized partition
    customized = {}
    customized["n_partition"] = n_partition
    customized["n_items"] = n_item
    customized["exe_time"] = exe_time


    # full result
    result = {"default": default, "customized": customized}

    #output result in json
    output_file_path = sys.argv[2]
    with open(output_file_path, 'w') as f:
        json.dump(result, f)