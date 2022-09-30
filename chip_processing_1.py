# coding:utf-8
import os.path as osp
import os
import sys
import argparse
import itertools
import numpy as np
from Bio import SeqIO
import pyBigWig


SEQ_LEN = 501
INDEX = ['chr' + str(i + 1) for i in range(23)]
INDEX[22] = 'chrX'
CHROM = {}
convscore1 = {}
convscore2 = {}


def data_load():
    with open('hg19/chromsize') as fp:
        for line in fp:
            line_split = line.strip().split()  # strip()函数用于删除空白符 而split()用于分割，默认空白符
            if line_split[0] not in INDEX:
                continue
            CHROM[line_split[0]] = int(line_split[1])  # key:chr1   value:249250621  "染色体号——染色体长度"键值对
    path = "conservation"
    files = os.listdir(path)
    for file in files:
        position = path + '/' + file
        with open(position, "r", encoding='utf-8') as f:
            data = f.readline().strip().split(' ')
            chrom_name = data[1][6:]
            start_pos = data[2][6:]
            convscore1[chrom_name] = int(start_pos)

            seq = []
            while True:
                line = f.readline()
                if not line:
                    break
                if len(line) > 10:
                    continue
                seq.append(float(line[:-1]))
            min_value = min(seq)
            max_value = max(seq)
            for idx in range(len(seq)):
                seq[idx] = (seq[idx] - min_value) / (max_value - min_value)
            convscore2[chrom_name] = seq


def one_hot(sequence_dict, chrom, start, end):
    seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 0, 1, 0],
                'C':[0, 1, 0, 0], 'T':[0, 0, 0, 1],
                'a':[1, 0, 0, 0], 'g':[0, 0, 1, 0],
                'c':[0, 1, 0, 0], 't':[0, 0, 0, 1]}
    temp = []
    seq = str(sequence_dict[chrom].seq[start:end])
    start_pos = convscore1[chrom]
    li = convscore2[chrom][start - start_pos: end - start_pos]
    index = 0
    for c in seq:
        t = []
        t.extend(seq_dict.get(c, [0, 0, 0, 0]))
        if index >= len(li):
            t.append(0)
        else:
            t.append(li[index])
        index += 1
        temp.append(t)
    return temp


def denselabel(data, pfmfile):
    """data: N*4*L, pfm: k*4"""
    pfm = []
    with open(pfmfile, 'r') as f:
        for line in f:
            line_split = line.strip().split()
            pfm.append([float(i) for i in line_split])
    pfm = np.asarray(pfm)
    pfm = pfm.transpose((1, 0))
    N, _, L = data.shape
    _, k = pfm.shape
    denselabels = []
    for i in range(N):
        data_row = data[:, [0, 1, 2, 3]][i]  # get one-hot information
        records = np.zeros(L-k+1)
        for j in range(L-k+1):
            records[j] = np.sum(data_row[:, j:(j+k)] * pfm)
        best_index = np.argmax(records)
        denselabel_row = np.zeros(L)
        denselabel_row[best_index:(best_index+k)] = 1.
        denselabels.append(denselabel_row)

    return np.asarray(denselabels)  # N*L


def pos_location(chr, start, end, resize_len):
    original_len = end - start
    if original_len < resize_len:
        start_update = start - np.ceil((resize_len - original_len) / 2)
    elif original_len > resize_len:
        start_update = start + np.ceil((original_len - resize_len) / 2)
    else:
        start_update = start

    end_update = start_update + resize_len
    if end_update > CHROM[chr]:
        end_update = CHROM[chr]
        start_update = end_update - resize_len
    return int(start_update), int(end_update)


def get_data(seqs_bed, sequence_dict, pfmfile):
    seqs = []  # encode
    lines = open(seqs_bed).readlines()
    index = list(range(len(lines)))
    # np.random.shuffle(index)
    for i in index:
        line_split = lines[i].strip().split()  # 第一个数据是染色体编号(chr1) 第二个数据是碱基序列的起始位置 第三个是结束位置
        chr = line_split[0]  # 染色体编号 chr1
        if chr not in INDEX:
            continue
        start, end = int(line_split[1]), int(line_split[2])  # 碱基序列起始和结束位点 878407	878870
        start_p, end_p = pos_location(chr, start, end, SEQ_LEN)  # 在这段序列上选取长度为SEQ_LEN的子序列
        seqs.append(one_hot(sequence_dict, chr, start_p, end_p))  # 将序列进行one-hot编码

    seqs = np.array(seqs, dtype=np.float32)
    seqs = seqs.transpose((0, 2, 1))

    labels_dense = denselabel(seqs, pfmfile)

    return seqs, labels_dense


def get_args():
    parser = argparse.ArgumentParser(description="pre-process data.")
    parser.add_argument("-d", dest="dir", type=str, default='')
    parser.add_argument("-n", dest="name", type=str, default='')

    return parser.parse_args()


def main():
    params = get_args()
    name = params.name
    data_dir = params.dir
    out_dir = osp.join(params.dir, 'data/')
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    else:
        print("{} data have been processed!".format(name))
        return
    data_load()
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open('hg19/hg19.fa'), 'fasta'))
    print('Experiment on %s dataset' % name)
    seqs_bed = data_dir + '/all_sort_merge.bed'
    pfmfile = data_dir + '/%s.txt' % name
    seqs, labels_dense = get_data(seqs_bed, sequence_dict, pfmfile)

    np.savez(out_dir+'%s_data.npz' % name, data=seqs, denselabel=labels_dense)


if __name__ == '__main__':
    main()
