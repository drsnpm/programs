import math
import csv
class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ""
        
def load_csv(filename):
    csv_file = csv.reader(open(filename, "r"));
    data = list(csv_file)
    head = data.pop(0)
    return data, head

def build_tree(data, head):
    lastcol = [row[-1] for row in data]
    if (len(set(lastcol))) == 1:
        node = Node("")
        node.answer = lastcol[0]
        return node
    gains = [compute_gain(data, col) for col in range(len(data[0])-1) ]
    split = gains.index(max(gains)) 
    node = Node(head[split]) 
    fea = head[:split] + head[split+1:]
    attr, dic = subtables(data, split, delete=True) 
    for x in range(len(attr)):
        child = build_tree(dic[attr[x]], fea) 
        node.children.append((attr[x], child))
    return node

def compute_gain(data, col):
    attValues, dic = subtables(data, col, delete=False)
    total_entropy = entropy([row[-1] for row in data])
    for x in range(len(attValues)):
        ratio = len(dic[attValues[x]]) / ( len(data) * 1.0)
        entro = entropy([row[-1] for row in dic[attValues[x]]]) 
        total_entropy -= ratio*entro
    return total_entropy

def subtables(data, col, delete): 
    dic = {}
    coldata = [ row[col] for row in data]
    attr = list(set(coldata)) 
    for k in attr:
        dic[k] = []
    for y in range(len(data)):
        key = data[y][col]
        if delete:
            del data[y][col]
        dic[key].append(data[y])
    return attr, dic

def entropy(S):
    attr = list(set(S))
    if len(attr) == 1: 
        return 0
    counts = [0,0] 
    for i in range(2):
        counts[i] = sum( [1 for x in S if attr[i] == x] ) / (len(S) * 1.0)
    sums = 0
    for cnt in counts:
        sums += -1 * cnt * math.log(cnt, 2)
    return sums

def print_tree(node, level):
    if node.answer != "":
        print("     "*level, node.answer) 
        return
    print("       "*level, node.attribute) 
    for i,j in node.children:
        print("     "*(level+1), i) 
        print_tree(j, level + 2)

def classify(node,x_test,head): 
    if node.answer != "":
        print(node.answer) 
        return
    pos = head.index(node.attribute)
    for i,j in node.children:
        if x_test[pos] == i: 
            classify(j,x_test,head)
            
data, head = load_csv("dat3.csv")
node = build_tree(data, head)
print("The decision tree for the dataset using ID3 algorithm is ") 
print_tree(node, 0)
data, head = load_csv("data3test.csv") 
for x_test in data:
    print("The test instance : ",x_test) 
    print("The predicted label : ", end="") 
    classify(node,x_test,head)
