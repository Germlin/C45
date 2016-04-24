# !/usr/bin/python
#  -*- coding: utf-8 -*-

import math
import argparse
import pygraphviz as pyg


class Node:
    """
    C45的决策树结点，每个结点需要包含以下信息：
    1. 这个结点可以使用哪些属性来分裂；
    2. 这个结点的子树包含了哪些样例；
    3. 这个结点本身是由父结点的哪个属性分裂来的，它的属性值是什么。
    """

    def __init__(self, select_row, attribute, parent_a, value):
        self.sample = select_row
        self.attribute = attribute
        self.parent_attr = parent_a
        self.value = value
        self.child = []


class Tree:
    """
    C45的决策树，有一个分裂函数，一个构造函数。
    分裂函数的参数是一个结点，这个结点要求至少有两个属性和，它计算出每一个可用属性的增益，用最高的属性进行分裂，
    构造子节点。
    构造函数，从根节点开始，除非classify出来的字典只有一个元素，或者已经没有属性
    可以用来分裂了。
    """

    def __init__(self, data):
        self.matrix = data
        self.row = len(data)
        self.col = len(data[0])
        self.root = Node(list(range(self.row)), list(range(self.col - 1)), self.col, 'root')
        self.build(self.root)

    def split(self, node):
        gain_max = 0
        gain_max_attr = 0
        gain_max_dict = {}
        res = []
        if len(node.attribute) == 0:
            return res
        for attr in node.attribute:
            t = self.entropy(node.sample)
            if t == 0:
                return res
            d = self.classify(node.sample, attr)
            c = self.conditional_entropy(node.sample, d)
            c_e = (t - c[0]) / c[1]
            if c_e > gain_max:
                gain_max = c_e
                gain_max_attr = attr
                gain_max_dict = d
        used_attr = node.attribute[:]
        used_attr.remove(gain_max_attr)
        for (k, v) in gain_max_dict.items():
            res.append(Node(v, used_attr, gain_max_attr, k))
        return res

    def entropy(self, index_list):
        """
        计算给定样本的熵。
        :param index_list: list类型，给出样本所在的行数。
        :return: 样本的熵。
        """
        sample = {}
        for index in index_list:
            key = self.matrix[index][self.col - 1]
            if key in sample:
                sample[key] += 1
            else:
                sample[key] = 1
        entropy_s = 0
        for k in sample:
            entropy_s += -(sample[k] / len(index_list)) * math.log2(sample[k] / len(index_list))
        return entropy_s

    def classify(self, select_row, column):
        res = {}
        for index in select_row:
            key = self.matrix[index][column]
            if key in res:
                res[key].append(index)
            else:
                res[key] = [index]
        return res

    def conditional_entropy(self, select_row, d):
        c_e = 0
        total = len(select_row)
        spilt_info = 0
        for k in d:
            c_e += (len(d[k]) / total) * self.entropy(d[k])
            spilt_info += -(len(d[k]) / total) * math.log2((len(d[k]) / total))
        return (c_e, spilt_info)

    def build(self, root):
        child = self.split(root)
        root.child = child
        if len(child) != 0:
            for i in child:
                self.build(i)

    def save(self, filename):
        g = pyg.AGraph(strict=False, directed=True)
        g.add_node(self.root.value)
        self._save(g, self.root)
        g.layout(prog='dot')
        g.draw(filename)
        print("The file is save to %s." % filename)

    def _save(self, graph, root):
        if root.child:
            for node in root.child:
                graph.add_node(node.value)
                graph.add_edge(root.value, node.value)
                self._save(graph, node)
        else:
            graph.add_node(self.matrix[root.sample[0]], label=self.matrix[root.sample[0]][self.col - 1], shape="box")
            graph.add_edge(root.value, self.matrix[root.sample[0]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='data', type=argparse.FileType('r'), default="data.txt")
    args = parser.parse_args()
    matrix = []
    lines = args.data.readlines()
    for line in lines:
        matrix.append(line.split())
    C45tree = Tree(matrix)
    C45tree.save("data.png")
