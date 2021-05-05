from random import shuffle

def is_ascii(s):
    return all((ord(c) < 128 and ord(c) != 8 and ord(c) != 26) for c in s)

def get_file_len(infile):
    with open(infile) as f:
        i = 0
        for i, l in enumerate(f):
            pass
    return i + 1

def clean_dataset(infile, outfile):
    read_file = open(infile, "r")
    write_file = open(outfile, "w")

    for line in read_file.readlines():
        if len(line) <= 32 and is_ascii(line):
            write_file.write(line)

    read_file.close()
    write_file.close()

def create_training_testing_set(infile, train_outfile, test_outfile, ratio):
    dataset_size = get_file_len(infile)
    # train_size = int(dataset_size * ratio)
    train_size = 100000
    f = open(infile, "r")
    train_file = open(train_outfile, "w")
    # test_file = open(test_outfile, "w")
    lines = f.readlines()
    shuffle(lines)

    train_dataset = lines[:train_size]
    # test_dataset = lines[train_size:]

    for line in train_dataset:
        train_file.write(line)

    # for line in test_dataset:
    #     test_file.write(line)

    f.close()
    train_file.close()
    # test_file.close()

create_training_testing_set("datasets/1M_rockyou_train.txt", "datasets/100K_rockyou_train.txt", "", 0.8)


    


