import numpy as np



# data = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
#
# print(data[:,[0]],type(data[:,[0]]),data[:,[0]].shape)

'''
seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 0, 1, 0],
                'C':[0, 1, 0, 0], 'T':[0, 0, 0, 1],
                'a':[1, 0, 0, 0], 'g':[0, 0, 1, 0],
                'c':[0, 1, 0, 0], 't':[0, 0, 0, 1]}
temp = []
seq = ['a','c','g','t']
l3 = [1,2,3,4]
l4 = [1,2,3,4]
index = 0
for c in seq:
    t = []
    t.extend(seq_dict.get(c, [0, 0, 0, 0]))
    t.append(l3[index])
    t.append(l4[index])
    print(t, len(t))
    temp.append(t)
    index += 1
print(temp)
'''

# path = "/home/sc3/xu/fcna/A549_2/ZBTB33/data/ZBTB33_data.npz"
# data = np.load(path)
# print(data["data"][0],data["denselabel"][0],data["data"].shape,data["denselabel"].shape)
#
# position = "/home/sc3/xu/fcna/convscore/chr1.phyloP100way.wigFix"
# with open(position, "r", encoding='utf-8') as f:
#     data = f.readline().strip().split(' ')
#     chrom_name = data[1][6:]
#     start_pos = data[2][6:]
#     print("chrom_name = {}, start_pos = {}".format(chrom_name, start_pos))
#     index = 0
#     while True:
#         line = f.readline()
#         if not line:
#             break
#         if index > 10:
#             break
#         print(float(line[:-1]))
#         index += 1

# data = np.load("A549_2/ATF3/data/ATF3_data.npz")
# print(data["data"][0], data["data"][1])


def writeFile(outdir, flag):
    out_f = open(outdir + '/{}_pfm.txt'.format(flag), 'w')
    out_f.write("MEME version 5.1.1\n\n")
    out_f.write("ALPHABET= ACGT\n\n")
    out_f.write("strands: + -\n\n")
    out_f.write("Background letter frequencies\n")
    out_f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
    out_f.close()


if __name__ == "__main__":
    writeFile("motifs/FCNA/ATF3", "all_test")

















