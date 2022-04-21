from datetime import datetime
import os
import shutil
import unittest

import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F

from preprocess import get_test_loader
from preprocess import UserRoundData
from decision_tree.homo.homo_decision_tree_client import HomoDecisionTreeClient
from decision_tree.homo.homo_decision_tree_arbiter import HomoDecisionTreeArbiter


class ParameterServer(HomoDecisionTreeArbiter):
    def __init__(self):
        super().__init__()
        self.workers = []
        self.global_bin_split_points = []
        self.feature_num = 23
        self.boosting_round = 50
        self.booster_dim = 1
        self.bin_num = 10
        self.learning_rate = 0.1
        self.tree_list = []

    def collect_worker(self, worker):
        self.workers.append(worker)

    def aggregate(self):
        print('start aggregate')
        # 基于Quantile sketch计算全局的候选节点
        global_quantile = self.workers[0].fit_get_quantile()
        for worker in self.workers[1:]:
            summary_list = worker.fit_get_quantile()
            assert len(global_quantile) == len(summary_list)
            for fid in range(len(global_quantile)):
                global_quantile[fid].merge(summary_list[fid])

        percent_value = 1.0 / self.bin_num

        percentile_rate = [i * percent_value for i in range(1, self.bin_num)]
        percentile_rate.append(1.0)
        for sum_obj in global_quantile:
            sum_obj.compress()
            split_point = []
            for percen_rate in percentile_rate:
                s_p = sum_obj.query(percen_rate)
                if s_p not in split_point:
                    split_point.append(s_p)
            self.global_bin_split_points.append(split_point)

        # 对所有worker进行初始化
        for worker in self.workers:
            worker.fit_init(self.global_bin_split_points)
        for epoch_idx in range(self.boosting_round):
            print('epoch{}'.format(epoch_idx))
            # 对worker的每个booster阶段进行初始化
            for worker in self.workers:
                worker.fit_booster_init()
            for class_idx in range(self.booster_dim):
                print('class{}'.format(class_idx))
                # 对所有label种类分别建立基决策树
                g_sum, h_sum = 0, 0
                for worker in self.workers:
                    # 获取各个worker当前epoch和class的根节点g h
                    g, h = worker.fit_send_g_h(class_idx)
                    g_sum += g
                    h_sum += h
                for worker in self.workers:
                    worker.fit_get_global_g_h(g_sum, h_sum)
                if self.max_split_nodes != 0 and self.max_split_nodes % 2 == 1:
                    self.max_split_nodes += 1

                tree_height = self.max_depth + 1  # non-leaf node height + 1 layer leaf
                for dep in range(tree_height):
                    if dep + 1 == tree_height:
                        for worker in self.workers:
                            self.tree_list.append(worker.fit_break())
                        break
                    split_info = []
                    cur_layer_node_num = self.workers[0].fit_cur_layer_node_num()
                    for worker in self.workers[1:]:
                        assert worker.fit_cur_layer_node_num() == cur_layer_node_num

                    layer_stored_hist = {}
                    # for batch_id, i in enumerate(range(0, cur_layer_node_num, self.max_split_nodes)):
                    left_node_histogram = self.workers[0].fit_batch_send_local_h(dep)
                    for worker in self.workers[1:]:
                        worker_loacl_h = worker.fit_batch_send_local_h(dep)
                        # TODO BUG FIX
                        for nid, node in enumerate(worker_loacl_h):
                            # left_node_histogram[nid].merge_hist(worker_loacl_h[nid])
                            feature_hist1 = left_node_histogram[nid].bag
                            assert feature_hist1 is left_node_histogram[nid].bag
                            feature_hist2 = worker_loacl_h[nid].bag
                            assert len(feature_hist1) == len(feature_hist2)
                            for j in range(len(feature_hist1)):
                                assert len(feature_hist1[j]) == len(feature_hist2[j])
                                for k in range(len(feature_hist1[j])):
                                    assert len(feature_hist1[j][k]) == 3
                                    feature_hist1[j][k][0] += feature_hist2[j][k][0]
                                    feature_hist1[j][k][1] += feature_hist2[j][k][1]
                                    feature_hist1[j][k][2] += feature_hist2[j][k][2]
                    all_histograms = self.histogram_subtraction(left_node_histogram, self.stored_histograms)
                    # store histogram
                    for hist in all_histograms:
                        layer_stored_hist[hist.hid] = hist

                    best_splits = self.federated_find_best_split(all_histograms, parallel_partitions=10)
                    split_info += best_splits

                    self.stored_histograms = layer_stored_hist
                    for worker in self.workers:
                        worker.get_split_info(split_info)
                for worker in self.workers:
                    # 将叶子节点的bid转化为实际类
                    worker.fit_convert()
                # update feature importance
                # for worker in self.workers:
                #     worker.fit_update_feature_importance()
                #
                # update predict score
                for worker in self.workers:
                    worker.fit_update_y_hat(class_idx, self.learning_rate)
                    self.tree_list.append(worker.fit_send_tree_list())

            # loss compute
            # local_loss = self.compute_loss(self.y_hat, self.y)
            # self.aggregator.send_local_loss(local_loss, self.data_bin.count(), suffix=(epoch_idx,))
        # print summary
        # self.set_summary(self.generate_summary())

    def predict_data(self, data):
        #        predict s = prediction.mapValues(lambda f: loss_method.predict(f).tolist())

        # tree_list = []
        # weight_list = []
        # for tree in tree_list:
        #     weight = tree.traverse_tree(data, tree.tree_node)
        #     weight_list.append(weight)

        return self.workers[0].predict(data, self.learning_rate, self.tree_list)


class FedAveragingGradsTestSuit(unittest.TestCase):
    RESULT_DIR = 'result'
    N_VALIDATION = 10000
    TEST_BASE_DIR = './tmp/'

    def setUp(self):
        self.seed = 0
        self.use_cuda = False
        self.batch_size = 64
        self.test_batch_size = 1000
        self.lr = 0.001
        self.n_max_rounds = 10
        self.log_interval = 10
        self.n_round_samples = 1600
        self.testbase = self.TEST_BASE_DIR
        self.testworkdir = os.path.join(self.testbase, 'competetion-test')

        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)
        print(self.testworkdir)
        # import pdb; pdb.set_trace()
        self.init_model_path = os.path.join(self.testworkdir, 'init_model.md')
        self.ps = ParameterServer()

        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)

        self.urd = UserRoundData()
        self.n_users = self.urd.n_users

    def _clear(self):
        shutil.rmtree(self.testworkdir)

    def tearDown(self):
        self._clear()

    def test_federated_averaging(self):

        for u in range(0, self.n_users):
            x, y = self.urd.round_data(user_idx=u)
            model = HomoDecisionTreeClient(data_bin=(x, y))
            self.ps.collect_worker(worker=model)

        self.ps.aggregate()

        #     if model is not None and r % 200 == 0:
        #         self.predict(model,
        #                      device,
        #                      self.urd.uniform_random_loader(self.N_VALIDATION),
        #                      prefix="Train")
        #         self.save_testdata_prediction(model=model, device=device)
        #
        # if model is not None:
        self.save_testdata_prediction()

    def save_prediction(self, predition):
        if isinstance(predition, (np.ndarray,)):
            predition = predition.reshape(-1).tolist()

        with open(os.path.join(self.RESULT_DIR, 'result.txt'), 'w') as fout:
            fout.writelines(os.linesep.join([str(n) for n in predition]))

    def save_testdata_prediction(self):
        loader = get_test_loader(batch_size=1000)
        with torch.no_grad():
            prediction = self.ps.predict_data(loader)
        self.save_prediction(prediction)

    def predict(self, model, device, test_loader, prefix=""):
        model.eval()
        test_loss = 0
        correct = 0
        prediction = []
        real = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(
                    output, target,
                    reduction='sum').item()  # sum up batch loss
                pred = output.argmax(
                    dim=1,
                    keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                prediction.extend(pred.reshape(-1).tolist())
                real.extend(target.reshape(-1).tolist())

        test_loss /= len(test_loader.dataset)
        acc = 100. * correct / len(test_loader.dataset)
        print(classification_report(real, prediction))
        print(
            '{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                prefix, test_loss, correct, len(test_loader.dataset), acc), )


def suite():
    suite = unittest.TestSuite()
    suite.addTest(FedAveragingGradsTestSuit('test_federated_averaging'))
    return suite


def main():
    runner = unittest.TextTestRunner()
    runner.run(suite())


if __name__ == '__main__':
    main()
