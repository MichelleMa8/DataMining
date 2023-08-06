from blackbox import BlackBox
import binascii
import sys, datetime, random

len_bit_array = 69997
random.seed(2)
hash_a = [random.randint(1,500) for i in range(100)]
hash_b = [random.randint(0,2000) for i in range(100)]

def myhashs(s):
    # hash function: (a * x + b) % len_bit_array, convert it into 16b
    result = []
    user_encode = int(binascii.hexlify(s.encode('utf8')),16)
    for i in range(len(hash_a)):
        encode = format((hash_a[i] * user_encode + hash_b[i]) % len_bit_array, '016b')
        result.append(encode)
    return result

def Flajolet_Martin(asks: list):
    # split into groups
    group = {i: {j: [] for j in range(group_hash)} for i in range(len(hash_a) // group_hash)}
    for s in asks:
        hash_res = myhashs(s)
        for i in range(len(hash_a) // group_hash):
            for j in range(group_hash):
                group[i][j].append(hash_res[i * group_hash + j])
                
    distinct = []
    for i in group.keys():
        group_dict = group[i]
        group_distinct = []
        for j in group_dict.keys():
            hash_val = group_dict[j]
            max_len = 0
            for val in hash_val:
                max_len = max(max_len, len(val) - len(val.rstrip("0")))
            group_distinct.append(2 ** max_len)  
            
        # average estimated distinct within a group
        distinct.append(sum(group_distinct)/len(group_distinct))
    
    distinct.sort()

    # mean estimated distinct among groups
    mid = len(distinct) // 2
    return distinct[mid]


if __name__ == '__main__':
    start = datetime.datetime.now()

    input_file_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    n_ask = int(sys.argv[3])
    output_file_path = sys.argv[4]

    window_len = 3
    sliding_interval = 2
    group_hash = 2
    
    # generate stream data
    bx = BlackBox()
    stream_users = {i : [] for i in range(n_ask)}
    for i in range(n_ask):
        stream_users[i] = bx.ask(input_file_path, stream_size)

    result = []
    stop = False
    for i in range(0, n_ask, sliding_interval):
        asks = []
        for j in range(window_len):
            if i + j < len(stream_users):
                asks += stream_users[i+j]  
            else:
                stop = True
        if not stop:
            estimate_distinct = Flajolet_Martin(asks)
            ground_truth = len(set(asks))
            result.append([ground_truth, estimate_distinct])
        else:
            break
    
    # write result
    output_file = open(output_file_path, 'w')
    output_file.write('Time,Ground Truth,Estimation\n')
    for i in range(len(result)):
        output_file.write(str(i) + ',' + str(result[i][0]) + ',' + str(round(result[i][1])) + '\n')
    output_file.close()
        
    end = datetime.datetime.now()
    print((end - start).seconds)