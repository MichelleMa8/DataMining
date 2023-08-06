from blackbox import BlackBox
import binascii
import sys, datetime

len_bit_array = 69997
hash_a = [1, 2, 3, 5, 7, 11, 13, 17, 19]
hash_b = [4831, 59, 1414, 3958, 3081, 8665, 1673, 1612, 38450, 2614, 36480]

def myhashs(s):
    # hash function: (a * x + b) % len_bit_array
    result = []
    user_encode = int(binascii.hexlify(s.encode('utf8')),16)
    for i in range(len(hash_a)):
        result.append((hash_a[i] * user_encode + hash_b[i]) % len_bit_array)
    return result

def bloom_filter(stream_users):
    global fp, ground_truth, bit_array
    for user in stream_users:
        hash_res = myhashs(user)
        count = 0
        for idx in hash_res:
            if bit_array[idx] == 0:
                bit_array[idx] = 1
                count += 1
        # detected repeat
        if count == 0:
            if user not in ground_truth:
                fp.add(user)
        ground_truth.add(user)
    fpr = len(fp) / len(ground_truth)
    return fpr

if __name__ == '__main__':
    start = datetime.datetime.now()
    
    input_file_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    n_ask = int(sys.argv[3])
    output_file_path = sys.argv[4]  

    bit_array = [0] * len_bit_array
    fp = set()
    ground_truth = set()


    bx = BlackBox()
    result = []
    for _ in range(n_ask):
        stream_users = bx.ask(input_file_path, stream_size)
        result.append(bloom_filter(stream_users))
    
    output_file = open(output_file_path, 'w')
    output_file.write('Time,FPR\n')
    for i in range(len(result)):
        output_file.write(str(i) + ',' + str(result[i]) + '\n')
    output_file.close()
        
    end = datetime.datetime.now()
    print((end - start).seconds)