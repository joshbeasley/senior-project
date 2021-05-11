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

def get_accuracy(test_file, data_file):
    test = open(test_file, "r")
    data = open(data_file, "r")

    test_lines = test.readlines()
    data_lines_full = data.readlines()
    nums = [1,10,100,1000,10000,100000]

    for num in nums:
        data_lines = data_lines_full[:num]
        matches = list(set(test_lines).intersection(data_lines))
        num_matches = len(matches)
        print(num_matches)

get_accuracy("datasets/myspace.txt", "passwordLSTM/output_full.txt")
    


