import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

data_folder = Path("C:/Users/localadmin/Desktop/Maxime_Bollengier/data_test/")
image = data_folder / "lung_020_0000.nii.gz"
image = nib.load(image)
#transform to canonical coordinates (RAS)
image = nib.as_closest_canonical(image)
image.set_data_dtype(np.uint8)
data = image.get_fdata()
image = data[0:10,0:10,0:10]

#image = np.arange(0,9).reshape(1,3,3)


def cord2idx(coor,Size):
    idx = Size[2]*Size[1] * coor[0] + Size[2]*coor[1] + coor[2]
    return idx
def img2graph (image):
    image = np.array(image)
    Size_x, Size_y , Size_z = image.shape
    Size_A = Size_x* Size_y * Size_z
    A = np.zeros([Size_A,Size_A], dtype=np.int8)
    nodes=np.zeros([2,Size_A])
    nodes[0,:]= np.arange(0,Size_A)
    for x in range (0,Size_x):
        for y in range(0, Size_y):
            for z in range(0, Size_z):
                #print([x,y,z],image[x,y,z],cord2idx([x,y,z],image.shape) )
                current = [x,y,z]
                current_idx = cord2idx(current,image.shape)
                #A[current_idx,current_idx] = 1
                for i in range(-1,2):
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            try :
                                nodes[1,cord2idx([x+i,y+j,z+k],image.shape)]=(image[x+i,y+j,z+k])
                                A[current_idx, cord2idx([x+i,y+j,z+k],image.shape)] = 1
                            except IndexError : #replace by a function "is_edge_voxel"
                                try:
                                    A[current_idx, cord2idx([x + i, y + j, z + k],image.shape)] = 0
                                except IndexError:
                                    continue

                                continue
    return (A.T,nodes)

def get_vertices (A):
    edges =[]
    for i in range (A.shape[1]):
        for j in range (i,A.shape[0]) :
            if A[i,j]==1 :
                edges.append((i,j))
                #edges.append((j, i)) #for two dir
    return edges

def showgraph(G):
    seed = 13648
    options = {
        'node_color': 'red',
        'node_size': 100,
        'width': 3,
        'with_labels' : True,
        'pos' : nx.spring_layout(G),
        'cmap' : plt.get_cmap('jet'),
    }
    print('number of nodes = ', G.number_of_nodes())
    print('number of edges = ', G.number_of_edges())
    pos = nx.spring_layout(G, seed=seed)
    nx.draw_networkx_nodes(G, pos, nodelist=nx.nodes(G) ,cmap=plt.get_cmap('jet'), node_size = 200)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=nx.edges(G), edge_color='g', arrows=True)
    #nx.draw(G, **options)
    plt.show()
#print('--------adjacency matrix\n',img2graph(image)[0])
#print('--------nodes feature\n', img2graph(image)[1])


G = nx.Graph()
A= img2graph(image)[0]
elist = get_vertices (A)
nlist = img2graph(image)[1][0]
print('edges_list\n', elist)
print('nodes_list\n', nlist)
G.add_edges_from(elist)
G.add_nodes_from(nlist)
showgraph(G)