import os
import argparse

class Data:
    def __init__(self, data_dir= None, is_visua = False):
        #load data
        self.is_visua = is_visua
        self.train_data = self.load_data(data_dir, "train")
        self.valid_data = self.load_data(data_dir, "valid")
        self.test_data = self.load_data(data_dir, "test")
        self.data = self.train_data + self.valid_data + self.test_data

        #get some statistics
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]
        self.timestamps = self.get_timestamps(self.data)

    def load_data(self, data_dir, data_type="train"):
        with open(os.path.join(data_dir, data_type+'.txt'), "r", encoding='utf-8') as f:
            data = f.readlines()
            if 'static' in data_dir:
                data = [line.strip().split("\t") + ['0'] for line in data]  # only cut by "\t", not by white space.
            else:
                data = [line.strip().split("\t") for line in data] #only cut by "\t", not by white space.
            if not self.is_visua:
                data += [[i[2], i[1]+"_reversed", i[0], i[3]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities

    def get_timestamps(self, data):
        timestamps = sorted(list(set([d[3] for d in data])))
        return timestamps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="GDELT_completion", nargs="?", help="Which dataset to use.")
    args = parser.parse_args()
    data_dir = "data/%s/" % args.dataset
    d = Data(data_dir=data_dir)
    timestamp_idxs = {d.timestamps[i]: i for i in range(len(d.timestamps))}

    file = open("test.txt", "w")
    for quadruple in d.test_data:
        file.write(str(quadruple[0]) + '\t' +  str(quadruple[1])  + '\t' + str(quadruple[2]) + "\t" + str(timestamp_idxs[quadruple[3]]) + "\t" + str(0) + '\n')
    file.close()