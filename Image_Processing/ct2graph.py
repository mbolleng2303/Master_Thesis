import cv2
from google.colab.patches import cv2_imshow
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from statistics import stdev, mean

# define globals required through out the whole program
edges = []  # containing all edge tuple
attrs = []  # countaining list of attribute of all nodes
graph_id = 1  # id of latest graph
node_id = 1  # id of latest node
graph_indicator = []  # containing graph-id for each node
node_labels = []  # containing labels for all node
graph_labels = []  # containing labels for all graph

# activity-label vs activity-name mapping (2-class)
activity_map = {}
activity_map[1] = 'COVID'
activity_map[2] = 'NON-COVID'


# z-score normalization
def normalize(arr):
    arr = np.array(arr)
    m = np.mean(arr)
    s = np.std(arr)
    return (arr - m) / s


# generate graph for a given edge-image file
def generate_graphs(filename, node_label, activity_map):
    print(" ... Reading image: " + filename + " ...")
    global node_id, edges, attrs, graph_id, node_labels, graph_indicator
    cnt = 0
    img = cv2.imread(filename)
    dim1, dim2, _ = img.shape
    attrs1 = []

    print("Image type: " + activity_map[node_label] + "\nPixel matrix is of: " + str(dim1) + "x" + str(dim2))
    img1 = img.copy()
    nodes = np.full((dim1, dim2), -1)
    edge = 0
    for i in range(dim1):
        for j in range(dim2):
            # considering pixel as node if pixel-value>=128
            b, _, _ = img[i][j]
            if (b >= 128):
                nodes[i][j] = node_id
                attrs1.append(b)
                graph_indicator.append(graph_id)
                node_labels.append([node_label, activity_map[node_label]])
                node_id += 1
                cnt += 1
            else:
                img1[i][j] = 0

    for i in range(dim1):
        for j in range(dim2):
            # forming edge between all adjacent pixels which are node
            if (nodes[i][j] != -1):
                li = max(0, i - 1)
                ri = min(i + 2, dim1)
                lj = max(0, j - 1)
                rj = min(j + 2, dim2)
                for i1 in range(li, ri):
                    for j1 in range(lj, rj):
                        if ((i1 != i or j1 != j) and (nodes[i1][j1] != -1)):
                            edges.append([nodes[i][j], nodes[i1][j1]])
                            edge += 1

    attrs1 = normalize(attrs1)
    attrs.extend(attrs1)
    del attrs1
    print("For given image nodes formed: " + str(cnt) + " edges formed: " + str(edge))
    if (cnt != 0): graph_id += 1


# generate graphs for all edge-image under given dir along with proper label
def generate_graph_with_labels(dirname, label, activity_map):
    print("\n... Reading Directory: " + dirname + " ...\n")
    global graph_labels
    filenames = glob.glob(dirname + '/*.png')
    for filename in filenames:
        generate_graphs(filename, label, activity_map)
        graph_labels.append([label, activity_map[label]])


# generate graphs for all directories
def process_graphs(covid_dir, ncovid_dir, activity_map):
    global node_labels, graph_labels
    generate_graph_with_labels(covid_dir, 1, activity_map)
    generate_graph_with_labels(ncovid_dir, 2, activity_map)
    print("Processing done")
    print("Total nodes formed: " + str(len(node_labels)) + "Total graphs formed: " + str(len(graph_labels)))


# working directories
covid_dir = '/content/drive/My Drive/Github_edge/Pretwitt/COVID'  # for covid
ncovid_dir = '/content/drive/My Drive/Github_edge/Pretwitt/NON-COVID'  # for non-covid

process_graphs(covid_dir, ncovid_dir, activity_map)

# check all the lengths of globals
# comment if not necessary
print(len(node_labels))
print(len(graph_labels))
print(len(edges))
print(len(attrs))

# create adjacency dataframe
df_A = pd.DataFrame(columns=["node-1", "node-2"], data=np.array(edges))
print("Shape of edge dataframe: " + str(df_A.shape))
print("\n--summary of dataframe--\n", df_A.head(50))

# create node label dataframe
df_node_label = pd.DataFrame(data=np.array(node_labels), columns=["label", "activity-name"])
print("shape of node-label dataframe: " + str(df_node_label.shape))
print("\n--summary of dataframe--\n", df_node_label.head(50))

# create graph label dataframe
df_graph_label = pd.DataFrame(data=np.array(graph_labels), columns=["label", "activity-name"])
print("shape of node-label dataframe: " + str(df_graph_label.shape))
print("\n--summary of dataframe--\n", df_graph_label.head(50))

# create node-attribute dataframe (normalized grayscale value)
df_node_attr = pd.DataFrame(data=np.array(attrs), columns=["gray-val"])
print("shape of node-attribute dataframe: " + str(df_node_attr.shape))
print("\n--summary of dataframe--\n", df_node_attr.head(50))

# create graph-indicator datframe
df_graph_indicator = pd.DataFrame(data=np.array(graph_indicator), columns=["graph-id"])
print("shape of graph-indicator dataframe: " + str(df_graph_indicator.shape))
print("\n--summary of dataframe--\n", df_graph_indicator.head(50))

# omit activity name later for graph-label and node-label
# since GIN model will only accept the label
df_node_label = df_node_label.drop(["activity-name"], axis=1)
print(df_node_label.head(50))

df_graph_label = df_graph_label.drop(["activity-name"], axis=1)
print(df_graph_label.head(50))


def save_dataframe_to_txt(df, filepath):
    df.to_csv(filepath, header=None, index=None, sep=',', mode='w')


# save all the dataframes to .txt file
# path name: .../GraphTrain/dataset/<dataset_name>/raw/<dataset_name>_<type>.txt
# <type>:
# A--> adjancency matrix
# graph_indicator--> graph-ids of all node
# graph_labels--> labels for all graph
# node_attributes--> attribute(s) for all node
# node_labels--> labels for all node


sourcepath = '/content/drive/My Drive/GraphTrain/dataset/Github_Pretwitt/raw'
save_dataframe_to_txt(df_A, sourcepath + '/Github_Pretwitt_A.txt')
save_dataframe_to_txt(df_graph_indicator, sourcepath + '/Github_Pretwitt_graph_indicator.txt')
save_dataframe_to_txt(df_graph_label, sourcepath + '/Github_Pretwitt_graph_labels.txt')
save_dataframe_to_txt(df_node_attr, sourcepath + '/Github_Pretwitt_node_attributes.txt')
save_dataframe_to_txt(df_node_label, sourcepath + '/Github_Pretwitt_node_labels.txt')