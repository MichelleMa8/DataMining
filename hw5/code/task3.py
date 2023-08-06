from blackbox import BlackBox
import sys, datetime, random

if __name__ == '__main__':
    start = datetime.datetime.now()
    
    input_file_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    n_ask = int(sys.argv[3])
    output_file_path = sys.argv[4]

    random.seed(553)
    bx = BlackBox()
    reservoir = []
    seqnum = stream_size

    output_file = open(output_file_path, 'w')
    output_file.write('seqnum,0_id,20_id,40_id,60_id,80_id\n')

    for i in range(n_ask):
        stream_users = bx.ask(input_file_path, stream_size)
        if i == 0:
            reservoir = stream_users[:]
            seqnum = 100
        else:
            for user in stream_users:
                seqnum += 1
                p = random.random()
                if p < stream_size / seqnum:
                    idx = random.randint(0, stream_size - 1)
                    reservoir[idx] = user
        output_file.write('{0},{1},{2},{3},{4},{5}\n'\
                          .format(seqnum, reservoir[0], reservoir[20], reservoir[40], reservoir[60], reservoir[80]))
    output_file.close()

    end = datetime.datetime.now()
    print((end - start).seconds)              