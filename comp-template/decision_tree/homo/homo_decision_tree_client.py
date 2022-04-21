import functools
from abc import ABC

from ..tree_core.criterion import XgboostCriterion
from ..tree_core.feature_histogram import FeatureHistogram
from ..tree_core.decision_tree import DecisionTree
from ..tree_core.node import Node
from ..tree_core.feature_histogram import HistogramBag
from ..tree_core.splitter import SplitInfo
from ..tree_core.quantile_summaries import quantile_summary_factory
import numpy as np
from typing import List


class NoneType(object):
    def __eq__(self, obj):
        return isinstance(obj, NoneType)


class Instance(object):
    """
    Instance object use in all algorithm module

    Parameters
    ----------
    inst_id : int, the id of the instance, reserved fields in this version

    weight: float, the weight of the instance

    feature : object, ndarray or SparseVector Object in this version

    label: None of float, data label

    """

    def __init__(self, inst_id=None, weight=1.0, features=None, label=None):
        self.inst_id = inst_id
        self.weight = weight
        self.features = features
        self.label = label

    def set_weight(self, weight=1.0):
        self.weight = weight

    def set_label(self, label=1):
        self.label = label

    def set_feature(self, features):
        self.features = features


# if loss_type == "cross_entropy":
#     if self.num_classes == 2:
#         loss_func = SigmoidBinaryCrossEntropyLoss()
#     else:
#         loss_func = SoftmaxCrossEntropyLoss()

class HomoDecisionTreeClient(DecisionTree):

    def __init__(self, data_bin=None, valid_feature: dict = None, ):

        super(HomoDecisionTreeClient, self).__init__()
        self.book = None
        self.init_score = None
        self.booster_dim = 2
        self.bin_num = 10
        self.loss_method = None
        self.y_hat = None
        self.y = np.array(data_bin[1])
        self.ATTACK_TYPES = {}
        self.dep = None
        self.table_with_assignment = None
        # self.tree_idx = None
        self.splitter = XgboostCriterion()
        self.data_bin = data_bin[0]
        self.g_h = None
        self.all_g_h = None
        self.bin_split_points = []
        self.feature_num = 23
        # self.epoch_idx = epoch_idx
        self.split_info = []
        self.agg_histograms = []
        # check max_split_nodes
        if self.max_split_nodes != 0 and self.max_split_nodes % 2 == 1:
            self.max_split_nodes += 1

        """
        initializing here
        """
        self.valid_features = valid_feature
        self.tree_node = []  # start from root node
        self.tree_node_num = 0
        self.cur_layer_node = []
        self.runtime_idx = 0
        self.feature_importance = {}
        self.federated_send = {}

    """
    Computing functions
    """

    def get_node_map(self, nodes: List[Node], left_node_only=True):
        node_map = {}
        idx = 0
        for node in nodes:
            if node.id != 0 and (not node.is_left_node and left_node_only):
                continue
            node_map[node.id] = idx
            idx += 1
        return node_map

    def get_grad_hess_sum(self, grad_and_hess_table):
        grad, hess = 0, 0
        for gh in grad_and_hess_table:
            grad += gh[0]
            hess += gh[1]
        return grad, hess

    def get_local_histogram(self, cur_to_split: List[Node], g_h, table_with_assign,
                            split_points, sparse_point, valid_feature):

        node_map = self.get_node_map(nodes=cur_to_split)
        histograms = FeatureHistogram.calculate_histogram(
            table_with_assign, g_h,
            split_points, sparse_point,
            valid_feature, node_map,
            self.use_missing)

        hist_bags = []
        for hist_list in histograms:
            hist_bags.append(HistogramBag(hist_list))

        return hist_bags

    def get_left_node_local_histogram(self, cur_nodes: List[Node], tree: List[Node], g_h, table_with_assign,
                                      split_points, feature_num, valid_feature):
        node_map = self.get_node_map(cur_nodes, left_node_only=True)

        histograms = FeatureHistogram.calculate_histogram(
            table_with_assign, g_h, split_points,
            feature_num, valid_feature, node_map,
            self.use_missing, self.zero_as_missing)
        hist_bags = []
        for hist_list in histograms:
            hist_bags.append(HistogramBag(hist_list))

        left_nodes = []
        for node in cur_nodes:
            if node.is_left_node or node.id == 0:
                left_nodes.append(node)

        # set histogram id and parent histogram id
        for node, hist_bag in zip(left_nodes, hist_bags):
            hist_bag.hid = node.id
            hist_bag.p_hid = node.parent_nodeid
        return hist_bags

    """
    Tree Updating
    """

    def update_tree(self, cur_to_split: List[Node], split_info: List[SplitInfo]):
        """
        update current tree structure
        ----------
        split_info
        """
        next_layer_node = []
        assert len(cur_to_split) == len(split_info)

        for idx in range(len(cur_to_split)):
            sum_grad = cur_to_split[idx].sum_grad
            sum_hess = cur_to_split[idx].sum_hess
            if split_info[idx].best_fid is None or split_info[idx].gain <= self.min_impurity_split + 1e-8:
                cur_to_split[idx].is_leaf = True
                self.tree_node.append(cur_to_split[idx])
                continue

            cur_to_split[idx].fid = split_info[idx].best_fid
            cur_to_split[idx].bid = split_info[idx].best_bid

            p_id = cur_to_split[idx].id
            l_id, r_id = self.tree_node_num + 1, self.tree_node_num + 2
            cur_to_split[idx].left_nodeid, cur_to_split[idx].right_nodeid = l_id, r_id
            self.tree_node_num += 2

            l_g, l_h = split_info[idx].sum_grad, split_info[idx].sum_hess

            # create new left node and new right node
            left_node = Node(id=l_id,
                             sum_grad=l_g,
                             sum_hess=l_h,
                             weight=self.splitter.node_weight(l_g, l_h),
                             parent_nodeid=p_id,
                             sibling_nodeid=r_id,
                             is_left_node=True)
            right_node = Node(id=r_id,
                              sum_grad=sum_grad - l_g,
                              sum_hess=sum_hess - l_h,
                              weight=self.splitter.node_weight(sum_grad - l_g, sum_hess - l_h),
                              parent_nodeid=p_id,
                              sibling_nodeid=l_id,
                              is_left_node=False)

            next_layer_node.append(left_node)
            next_layer_node.append(right_node)
            self.tree_node.append(cur_to_split[idx])

            self.update_feature_importance(split_info[idx], record_site_name=False)

        return next_layer_node

    @staticmethod
    def assign_a_instance(row, tree: List[Node], bin_split_points, use_missing, use_zero_as_missing):
        leaf_status, nodeid = row[1]
        node = tree[nodeid]
        if node.is_leaf:
            return node.weight

        fid = node.fid
        bid = node.bid
        record_bid = 0
        if row[0][fid] > bin_split_points[fid][len(bin_split_points[fid]) - 1]:
            record_bid = len(bin_split_points[fid]) - 1
        else:
            for bi in range(len(bin_split_points[fid]) - 1):
                if row[0][fid] <= bin_split_points[fid][bi + 1]:
                    record_bid = bi
                    break

        if record_bid <= bid:
            return 1, tree[nodeid].left_nodeid
        else:
            return 1, tree[nodeid].right_nodeid

    def assign_instances_to_new_node(self, table_with_assignment, tree_node: List[Node]):
        assign_method = functools.partial(self.assign_a_instance, tree=tree_node,
                                          bin_split_points=self.bin_split_points, use_missing=self.use_missing,
                                          use_zero_as_missing=self.zero_as_missing)
        assign_result = []
        data_bin = []
        g_h = []
        result_index = 0
        for i in range(len(table_with_assignment[0])):
            result = assign_method((table_with_assignment[0][i], table_with_assignment[1][i]))
            if isinstance(result, tuple):
                assign_result.append(result)
                data_bin.append(table_with_assignment[0][i])
                g_h.append(self.g_h[i])
                # TODO check
                self.book[result_index] = self.book[i]
                result_index += 1
            else:
                assert self.sample_weights[self.book[i]] == 0
                self.sample_weights[self.book[i]] = result
        # leaf_val = assign_result.filter(lambda key, value: isinstance(value, tuple) is False)
        # assign_result = assign_result.subtractByKey(leaf_val)

        return (data_bin, assign_result), g_h

    """
    Pre/Post process
    """

    def get_feature_importance(self):
        return self.feature_importance

    def convert_bin_to_real(self):
        """
        convert current bid in tree nodes to real value
        """
        for node in self.tree_node:
            if not node.is_leaf:
                node.bid = self.bin_split_points[node.fid][node.bid]

    # def assign_instance_to_root_node(self, data_bin, root_node_id):
    #     return data_bin.mapValues(lambda inst: (1, root_node_id))

    """
    Fit & Predict
    """

    def fit_get_quantile(self):
        summary_list = []
        for i in range(self.feature_num):
            summary_list.append(quantile_summary_factory(False))
        for rid in range(len(self.data_bin)):
            for fid in range(len(self.data_bin[rid])):
                summary_list[fid].insert(self.data_bin[rid][fid])
        for sum_obj in summary_list:
            sum_obj.compress()
        return summary_list

    def fit_init(self, bin_split_points):
        self.bin_split_points = bin_split_points
        from ..cross_entropy import SigmoidBinaryCrossEntropyLoss
        self.loss_method = SigmoidBinaryCrossEntropyLoss()
        self.y_hat, self.init_score = self.loss_method.initialize(self.y)

    def fit_booster_init(self):
        print(self.y_hat[1])
        self.all_g_h = [(self.loss_method.compute_grad(self.y[i], self.loss_method.predict(self.y_hat[i])),
                         self.loss_method.compute_hess(self.y[i], self.loss_method.predict(self.y_hat[i]))) for i
                        in range(self.y.size)]

    def fit_send_g_h(self, class_idx):  # for class_idx in range(self.booster_dim)
        self.g_h = [(self.all_g_h[i][0][class_idx], self.all_g_h[i][1][class_idx]) for i in range(self.y.size)]
        self.book = {i: i for i in range(len(self.data_bin))}
        self.sample_weights = [0] * len(self.data_bin)
        self.tree_node_num = 0
        self.inst2node_idx = [(1, 0)] * len(self.data_bin)
        self.table_with_assignment = (self.data_bin, self.inst2node_idx)
        self.tree_node = []
        return self.get_grad_hess_sum(self.g_h)

    def fit_get_global_g_h(self, global_g_sum, global_h_sum):
        root_node = Node(id=0, sum_grad=global_g_sum, sum_hess=global_h_sum, weight=self.splitter.node_weight(
            global_g_sum, global_h_sum))
        self.cur_layer_node = [root_node]

    def fit_break(self):
        for node in self.cur_layer_node:
            node.is_leaf = True
            self.tree_node.append(node)
        for i in range(len(self.table_with_assignment[0])):
            assert self.sample_weights[self.book[i]] == 0
            self.sample_weights[self.book[i]] = self.tree_node[self.table_with_assignment[1][i][1]].weight
        assert len(self.sample_weights) == len(self.data_bin)
        # for i in range(len(self.sample_weights)):
        #     assert self.sample_weights[i] != 0
        return self.tree_node

    def fit_cur_layer_node_num(self):
        # self.split_info, self.agg_histograms = [], []
        self.split_info = []
        return len(self.cur_layer_node)

    def fit_batch_send_local_h(self, i):
        cur_to_split = self.cur_layer_node
        node_map = self.get_node_map(nodes=cur_to_split)
        print('node map is {}'.format(node_map))
        local_histogram = self.get_left_node_local_histogram(
            cur_nodes=cur_to_split,
            tree=self.tree_node,
            g_h=self.g_h,
            table_with_assign=self.table_with_assignment,
            split_points=self.bin_split_points,
            feature_num=self.feature_num,
            valid_feature=self.valid_features
        )
        # self.agg_histograms += local_histogram
        return local_histogram

    def get_split_info(self, split_info):
        new_layer_node = self.update_tree(self.cur_layer_node, split_info)
        self.cur_layer_node = new_layer_node
        self.table_with_assignment, self.g_h = self.assign_instances_to_new_node(self.table_with_assignment,
                                                                                 self.tree_node)

    def fit_convert(self):
        self.convert_bin_to_real()

    def fit_update_y_hat(self, class_idx, lr):
        cur_sample_weight = self.get_sample_weights()
        for index, hat in enumerate(self.y_hat):
            hat[class_idx] += cur_sample_weight[index] * lr
        print(self.y_hat)

    def fit_send_tree_list(self):
        return self.tree_node

    def traverse_tree(self, data_inst, tree: List[Node]):

        nid = 0  # root node id
        while True:
            if tree[nid].is_leaf:
                return tree[nid].weight

            cur_node = tree[nid]
            fid, bid = cur_node.fid, cur_node.bid

            record_bid = 0
            if data_inst[fid] > self.bin_split_points[fid][len(self.bin_split_points[fid]) - 1]:
                record_bid = len(self.bin_split_points[fid]) - 1
            else:
                for bi in range(len(self.bin_split_points[fid]) - 1):
                    if data_inst[fid] <= self.bin_split_points[fid][bi + 1]:
                        record_bid = bi
                        break
            if record_bid <= bid + 1e-8:
                nid = tree[nid].left_nodeid
            else:
                nid = tree[nid].right_nodeid

    def predict(self, data_inst, lr, tree_list):
        predicts = []
        for i, record in enumerate(data_inst):
            weight_list = []
            for tree in tree_list:
                weight_list.append(self.traverse_tree(record, tree,))
            weights = np.array(weight_list)
            weights = weights.reshape((-1, 1))
            # if i == 0:
                # print(weights)
                # print(np.sum(weights * lr, axis=0) + self.get_init_score())
            predicts_score = self.loss_method.predict(np.sum(weights * lr, axis=0) + self.get_init_score())
            predicts.append(predicts_score)
        return predicts

    def get_init_score(self):
        return self.init_score
