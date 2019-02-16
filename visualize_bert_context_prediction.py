import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
graph = nx.MultiDiGraph()
lst = list()
with open('predict.txt', 'r') as f:
    lst = f.readlines()

sen_dict = dict()

def get_id(sen):
    if sen not in sen_dict:
        sen_dict[sen] = len(sen_dict)
    return sen_dict[sen]

for line in lst:
    slots = line.strip().split('\t')
    slots = [slot.strip() for slot in slots]
    label = int(slots[0])
    predict = int(slots[1])
    confidence = float(slots[2])
    left = slots[3]
    right = slots[4]
    left_id = get_id(left)
    right_id = get_id(right)
    if predict == 1:
        graph.add_edge(left_id, right_id, rel="next", color='g')
    elif predict == 2:
        graph.add_edge(left_id, right_id, rel="if", color='r')

nx.draw(graph, with_labels=True)
plt.show()
nx.write_graphml(graph, "test.graphml")