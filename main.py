import time
from collections import defaultdict
from load_data import Data
import argparse
import torch.nn as nn
from rsgd import *
import datetime
import os
from model import *
class Experiment:
    def __init__(self, curvatures_fixed, learning_rate=[], dim=[], nneg=50,
                 num_iterations=500, batch_size=128, batch_size_eval = 256, lr_cur = 0.01, use_cosh = False,
                 time_rescale = 364, dropout = 0, tid_nneg = True, vmax = 1, resume_file = ""):
        self.learning_rate = learning_rate
        self.dim = dim
        self.nneg = nneg
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.lr_cur = lr_cur
        self.curvatures_fixed = curvatures_fixed
        self.use_cosh = use_cosh
        self.time_rescale = time_rescale
        self.dropout = dropout
        self.tid_nneg = tid_nneg
        self.vmax=vmax
        self.resume_file = resume_file
        self.softplus = nn.Softplus()

    def get_data_idxs(self, data):
        data_idxs = [[self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]],
                      self.timestamp_idxs[data[i][3]]] for i in range(len(data))]
        return data_idxs
    
    def get_ept_vocab(self, data, idxs):
        er_vocab = defaultdict(list)
        for quadruple in data:
            er_vocab[(quadruple[idxs[0]], quadruple[idxs[1]], quadruple[idxs[3]])].append(int(quadruple[idxs[2]]))
        return er_vocab

    def get_ep_vocab(self, data, idxs):
        er_vocab = defaultdict(list)
        for quadruple in data:
            er_vocab[(quadruple[idxs[0]], quadruple[idxs[1]])].append(int(quadruple[idxs[2]]))
        return er_vocab

    def batch_evaluate(self, model, d, eval_data):
        hits_raw_ob = []
        hits_tid_ob = []
        ranks_raw_ob = []
        ranks_tid_ob = []

        for i in range(10):
            hits_raw_ob.append([])
            hits_tid_ob.append([])

        eval_data_idxs = self.get_data_idxs(eval_data)
        print("Number of data points: %d" % len(eval_data_idxs))
        sr_vocab_eval = self.get_ep_vocab(self.get_data_idxs(d.data), [0, 1, 2, 3])

        for j in range(0, len(eval_data_idxs), self.batch_size_eval):
            data_batch = np.array(eval_data_idxs[j:j+self.batch_size_eval])


            r_idx = to_device(torch.tensor(np.tile(np.array([data_batch[:, 1]]).T, (1, len(d.entities)))))
            t = to_device(torch.tensor(np.tile(np.array([data_batch[:, 3]]).T, (1, len(d.entities))), dtype=torch.double) / self.time_rescale)
            e1_idx = to_device(torch.tensor(np.tile(np.array([data_batch[:, 0]]).T, (1, len(d.entities)))))
            e2_idx_gt = to_device(torch.tensor(np.array(data_batch[:, 2])))
            e2_idx_cand = to_device(torch.tensor(np.tile(np.array([range(len(d.entities))]), (e1_idx.shape[0], 1))))  # batch_size_eval * num_entities
            # rank all entity candidates w.r.t their scores
            predictions_ob = 0
            for component in model:
                predictions_ob += component.forward(e1_idx, r_idx, e2_idx_cand, t)

            for i in range(data_batch.shape[0]):
                data_point = data_batch[i, :]
                filt_tid = sr_vocab_eval[(data_point[0], data_point[1])]
                target_value = predictions_ob[i, e2_idx_gt[i]].item()
                ##raw results
                predictions_ob[i, data_point[0]] = -np.Inf  # no self-loop
                rank_raw = (predictions_ob[i, :] > target_value).sum().item()
                ranks_raw_ob.append(rank_raw + 1)
                for hits_level in [0, 2, 9]:
                    if rank_raw <= hits_level:
                        hits_raw_ob[hits_level].append(1.0)
                    else:
                        hits_raw_ob[hits_level].append(0.0)

                ##filtering
                predictions_ob[i, filt_tid] = -np.Inf
                predictions_ob[i, e2_idx_gt[i]] = target_value
                rank_tid = (predictions_ob[i, :] > target_value).sum().item()
                ranks_tid_ob.append(rank_tid + 1)

                for hits_level in [0, 2, 9]:
                    if rank_tid <= hits_level:
                        hits_tid_ob[hits_level].append(1.0)
                    else:
                        hits_tid_ob[hits_level].append(0.0)
            if not j%self.batch_size_eval:
                print("Evaluated sample number: " + str(j))

        for i in range(len(model)):
            curvature = model[i].curvature.data
            print('current curvature of model ' + str(i) + ': ' + str(curvature) + '\n')

        #######raw
        hits10_raw_ob = np.mean(hits_raw_ob[9])
        hits3_raw_ob = np.mean(hits_raw_ob[2])
        hits1_raw_ob = np.mean(hits_raw_ob[0])
        mr_raw_ob = np.mean(ranks_raw_ob)
        mrr_raw_ob = np.mean(1. / np.array(ranks_raw_ob))

        print('Hits @10 raw_ob: {0}'.format(hits10_raw_ob))
        print('Hits @3 raw_ob: {0}'.format(hits3_raw_ob))
        print('Hits @1 raw_ob: {0}'.format(hits1_raw_ob))
        print('Mean rank raw_ob: {0}'.format(mr_raw_ob))
        print('Mean reciprocal rank raw_ob: {0}'.format(mrr_raw_ob))

        ######time independent
        hits10_tid_ob = np.mean(hits_tid_ob[9])
        hits3_tid_ob = np.mean(hits_tid_ob[2])
        hits1_tid_ob = np.mean(hits_tid_ob[0])
        mr_tid_ob = np.mean(ranks_tid_ob)
        mrr_tid_ob = np.mean(1. / np.array(ranks_tid_ob))

        print('Hits @10 tid_ob: {0}'.format(hits10_tid_ob))
        print('Hits @3 tid_ob: {0}'.format(hits3_tid_ob))
        print('Hits @1 tid_ob: {0}'.format(hits1_tid_ob))
        print('Mean rank tid_ob: {0}'.format(mr_tid_ob))
        print('Mean reciprocal rank tid_ob: {0}'.format(mrr_tid_ob))

        return mrr_tid_ob, hits1_tid_ob, hits3_tid_ob, hits10_tid_ob, \
               mrr_raw_ob, hits1_raw_ob, hits3_raw_ob, hits10_raw_ob


    def train_and_eval(self):
        model = []
        for i, curvature in enumerate(self.curvatures_fixed):
            if curvature == 0:
                model.append(DyERNIE_E(d, self.dim[i], self.learning_rate[i], use_cosh = self.use_cosh, dropout=self.dropout,))
            elif curvature > 0:
               model.append(DyERNIE_S(d, self.dim[i], self.learning_rate[i], fixed_c=curvature))
            else:
                model.append(DyERNIE_P(d, self.dim[i], self.learning_rate[i], use_cosh=self.use_cosh, dropout=self.dropout,
                              vmax=self.vmax))

        opts = []
        for i in range(len(model)):
            component = model[i]
            param_names = [name for name, param in component.named_parameters()]
            opt = RiemannianSGD(component.parameters(), param_names=param_names) #lr=self.learning_rate,
            opts.append(opt)

        if self.resume_file:
            print("Loading Checkpoint from {}...".format(self.resume_file))
            if torch.cuda.is_available():
                checkpoint = torch.load(self.resume_file)
            else:
                checkpoint = torch.load(self.resume_file, map_location = torch.device('cpu'))
            it_start = checkpoint["epoch"]
            self.entity_idxs = checkpoint["entity_idxs"]
            self.relation_idxs = checkpoint["relation_idxs"]
            self.timestamp_idxs = checkpoint["timestamp_idxs"]
            main_dirName = checkpoint["main_dirName"]
            for i, component in enumerate(model):
                component.load_state_dict(checkpoint["model"][i])
            for i, opt in enumerate(opts):
                opt.load_state_dict(checkpoint["optimizer_state_dict"][i])
        else:
            it_start = 1
            self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
            self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}
            self.timestamp_idxs = {d.timestamps[i]: i for i in range(len(d.timestamps))}
            # save models
            now = datetime.datetime.now()
            dt_string = now.strftime("%d-%m-%Y,%H:%M:%S")
            main_dirName = os.path.join(args.save_dir, dt_string)
            if not os.path.exists(main_dirName):
                os.makedirs(main_dirName)

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        # get static/dynamic valid objects for each (subject, predicate, timestamp)
        if self.tid_nneg:
            sr_vocab_train = self.get_ep_vocab(train_data_idxs, [0, 1, 2])
        else:
            srt_vocab_train = self.get_ept_vocab(train_data_idxs, [0, 1, 2, 3])

        if torch.cuda.is_available():
            for component in model:
                component.cuda()

        print("Starting training...")
        best_it = 0
        best_mrr = 0
        best_hits1 = 0
        best_hits3 = 0
        best_hits10 = 0
        best_curvature = None
        nn_loss = nn.BCEWithLogitsLoss()

        # create a txt file to write training procedure
        file_training_path = os.path.join(main_dirName, "training_record.txt")
        if not os.path.isfile(file_training_path):
            file_training = open(file_training_path, "w")
            file_training.write("Training Configuration: \n")
            for arg in vars(args):
                file_training.write(arg + ': ' + str(getattr(args, arg)) + '\n')

            file_training.write("Number of training data points: %d \n" % len(train_data_idxs))
            file_training.write("Training Start \n")
            file_training.write("===============================\n")
        else:
            file_training = open(file_training_path, "a")
            file_training.write("===============================\n")
            file_training.write("Resume from epoch: " + str(it_start) + "\n")

        for it in range(it_start, self.num_iterations+1):
            start_train = time.time()
            for component in model:
                component.train()
            if not args.only_eval:
                losses = []
                np.random.shuffle(train_data_idxs)

                for j in range(0, len(train_data_idxs), self.batch_size):
                    data_batch = np.array(train_data_idxs[j:j+self.batch_size])
                    loss = 0
                    for opt in opts:
                        opt.zero_grad()

                    if self.tid_nneg:
                        negsamples = getBatch_filter_all_static(train_data_idxs[j:j + self.batch_size],
                                                                len(d.entities),
                                                                sr_vocab_train, or_vocab=None,
                                                                corrupt_head=False,
                                                                mult_num=self.nneg)  # batch_size * nneg * 4
                    else:
                        negsamples = getBatch_filter_all_dyn(train_data_idxs[j:j + self.batch_size],
                                                             len(d.entities),
                                                             srt_vocab_train, ort_vocab=None,
                                                             corrupt_head=False,
                                                             mult_num=self.nneg)  # batch_size * nneg * 4

                    e1_idx = torch.tensor(np.concatenate((np.array([data_batch[:, 0]]).T, negsamples[:, :, 0]), axis = 1))
                    r_idx = torch.tensor(np.concatenate((np.array([data_batch[:, 1]]).T, negsamples[:, :, 1]), axis = 1))
                    e2_idx = torch.tensor(np.concatenate((np.array([data_batch[:, 2]]).T, negsamples[:, :, 2]), axis = 1))
                    t = torch.tensor(np.concatenate((np.array([data_batch[:, 3]]).T, negsamples[:, :, 3]), axis = 1), dtype=torch.double)/(self.time_rescale)
                    targets = np.zeros(e1_idx.shape)
                    targets[:, 0] = 1
                    targets = torch.DoubleTensor(targets)

                    if torch.cuda.is_available():
                        e1_idx = e1_idx.cuda()
                        r_idx = r_idx.cuda()
                        e2_idx = e2_idx.cuda()
                        t = t.cuda()
                        targets = targets.cuda()

                    #Make Predictions
                    predictions = 0
                    for component in model:
                        predictions += component.forward(e1_idx, r_idx, e2_idx, t)
                    loss += nn_loss(predictions, targets)

                    loss.backward()
                    losses.append(loss.item())

                    for i in range(len(opts)):
                        opt = opts[i]
                        curvature = model[i].curvature
                        opt.step(curvature, model[i].name, self.lr_cur, lr=model[i].learning_rate)

                file_training = open(file_training_path, "a")
                print("epoch: " + str(it))
                print("consuming time of current epoch: " + str(time.time()-start_train))
                print("loss: " + str(np.mean(losses)))
                file_training.write('epoch: ' + str(it) + '\n')
                file_training.write('consuming time of this iter: ' + str(time.time() - start_train) + '\n')
                file_training.write('loss: ' + str(np.mean(losses)) + '\n')

                save(os.path.join(main_dirName, "checkpoint_latest.pt"), file_training, model, args,
                     opts, it, self.entity_idxs, self.relation_idxs, self.timestamp_idxs, main_dirName)

            for component in model:
                component.eval()

            with torch.no_grad():
                eval_data = d.valid_data if args.eval_data == "valid_data" else d.test_data
                if not it%args.eval_ep:
                    print("Evaluation:")
                    file_training = open(file_training_path, "a")
                    file_training.write("Evaluation: \n")
                    mrr_tid, hits1_tid, hits3_tid, hits10_tid, \
                    mrr_raw, hits1_raw, hits3_raw, hits10_raw = self.batch_evaluate(model, d, eval_data)

                    file_training.write('mrr(raw): ' + str(mrr_raw) + '\n')
                    file_training.write('hits1(raw): ' + str(hits1_raw) + '\n')
                    file_training.write('hits3(raw): ' + str(hits3_raw) + '\n')
                    file_training.write('hits10(raw): ' + str(hits10_raw) + '\n')

                    file_training.write('mrr(time-independent filtering): ' + str(mrr_tid) + '\n')
                    file_training.write('hits1(time-independent filtering): ' + str(hits1_tid) + '\n')
                    file_training.write('hits3(time-independent filtering): ' + str(hits3_tid) + '\n')
                    file_training.write('hits10(time-independent filtering): ' + str(hits10_tid) + '\n')

                    for m, component in enumerate(model):
                        curvature = component.curvature.data
                        file_training.write('current curvature of component ' + str(m) + ': ' + str(curvature) + '\n')


                    ##################print best metrics until iteration it
                    if mrr_tid > best_mrr:
                        best_mrr = mrr_tid
                        best_hits1 = hits1_tid
                        best_hits3 = hits3_tid
                        best_hits10 = hits10_tid
                        best_it = it
                        best_curvature = [model[i].curvature.data for i in range(len(model))]

                    print('best mrr tid: ' + str(best_mrr))
                    print('best hits1 tid: ' + str(best_hits1))
                    print('best hits3 tid: ' + str(best_hits3))
                    print('best hits10 tid: ' + str(best_hits10))
                    print('best epoch: ' + str(best_it))
                    print('best curvature: ' + str(best_curvature) + '\n')

                    file_training.write('best mrr tid: ' + str(best_mrr) + '\n')
                    file_training.write('best hits1 tid: ' + str(best_hits1) + '\n')
                    file_training.write('best hits3 tid: ' + str(best_hits3) + '\n')
                    file_training.write('best hits10 tid: ' + str(best_hits10) + '\n')
                    file_training.write('best epoch: ' + str(best_it) + '\n')
                    file_training.write('best curvature: ' + str(best_curvature) + '\n')
                    file_training.write('\n')

                    save(os.path.join(main_dirName, "checkpoint_{:05d}.pt".format(it)), file_training, model, args,
                         opts, it, self.entity_idxs, self.relation_idxs, self.timestamp_idxs, main_dirName)

            file_training.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_eval", type=str2bool, default=False, help="whethre only evaluation")
    parser.add_argument('--tid_nneg', type=str2bool, default=True,
                        help='whether use time-independent negative sampling or tim-dependent negative sampling')
    parser.add_argument("--dataset", type=str, default="ICEWS14_completion", nargs="?",
                    help="Which dataset to use.")
    parser.add_argument("--num_iterations", type=int, default=1000, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=512, nargs="?",
                    help="Batch size.")
    parser.add_argument("--batch_size_eva", type=int, default=32, nargs="?",
                        help="Batch size in evalutaion phasing.")
    parser.add_argument("--nneg", type=int, default=500, nargs="?",
                    help="Number of negative samples.")
    parser.add_argument("--time_rescale", type=float, default=364, nargs="?",
                        help="scaling parameter for timestamp unit.")
    parser.add_argument("--save_dir", type=str, default="saved")
    parser.add_argument("--eval_ep", type=int, default=1, help='evaluation on the dataset per eval epochs.')
    parser.add_argument("--curvatures_fixed", nargs='+', type=float, default=[-0.346, -0.137, -0.855])
    parser.add_argument("--lr", nargs='+', type=float, default=[50, 50, 50],
                        help="Learning rate in each model")
    parser.add_argument("--dim", nargs='+', type=int, default=[20, 21, 59],
                        help="Embedding dimensionality.")
    parser.add_argument("--vmax", type=float, default=1., help="maximum velocity")
    parser.add_argument("--resume_file", type=str, default= "", help="model file to resume")
    parser.add_argument("--eval_data", type=str, default = "valid_data", help="set eval data to be test data or valid data")
    parser.add_argument("--use_cosh", type=str2bool, default=False, help="whether use cosh in score function")
    parser.add_argument("--lr_cur", type=float, default=0.01, nargs="?", help="Learning rate for curvature.")
    parser.add_argument("--dropout", type=float, default=0, help="non_zero part of time velocity vector")
    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True

    seed = 40
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)

    d = Data(data_dir=data_dir)
    experiment = Experiment(args.curvatures_fixed, learning_rate=args.lr, batch_size=args.batch_size,
                            num_iterations=args.num_iterations, dim=args.dim, nneg=args.nneg, batch_size_eval = args.batch_size_eva,
                            lr_cur = args.lr_cur,use_cosh = args.use_cosh, time_rescale = args.time_rescale, dropout = args.dropout,
                            tid_nneg=args.tid_nneg, vmax=args.vmax, resume_file = args.resume_file)
    experiment.train_and_eval()
