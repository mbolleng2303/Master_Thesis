import pickle
import time
import json
from scipy import sparse as sp
from batchgenerators.utilities.file_and_folder_operations import join
import dgl

from Image_Processing.preprocessing import normalize, normalize1
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
os.environ['DGLBACKEND'] = 'pytorch'
from dgl import DGLGraph

def create_lists_from_splitted_dataset(base_folder_splitted):
    lists = []
    json_file = join(base_folder_splitted, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d['training']
    num_modalities = len(d['modality'].keys())
    for tr in training_files:
        cur_pat = []
        for mod in range(num_modalities):
            cur_pat.append(join(base_folder_splitted, "imagesTr", tr['image'].split("/")[-1][:-7] +
                                "_%04.0d.nii.gz" % mod))
        cur_pat.append(join(base_folder_splitted, "labelsTr", tr['label'].split("/")[-1]))
        lists.append(cur_pat)
    return lists


def cord2idx(coor, size):
    idx = size[2] * size[1] * coor[0] + size[2] * coor[1] + coor[2]
    return idx


def img2graph(image):
    image = np.array(image)
    Size_x, Size_y , Size_z = image.shape
    Size_A = Size_x* Size_y * Size_z
    A = np.zeros([Size_A,Size_A])
    nodes=torch.zeros(Size_A)
    #nodes[0,:]= np.arange(0,Size_A)
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
                                nodes[cord2idx([x+i,y+j,z+k],image.shape)]= (image[x+i,y+j,z+k])
                                A[current_idx, cord2idx([x+i,y+j,z+k],image.shape)] = 1
                            except IndexError : # TODO : replace by a function "is_edge_voxel"
                                try:
                                    A[current_idx, cord2idx([x + i, y + j, z + k],image.shape)] = 0
                                except IndexError:
                                    continue

                                continue
    return (A.T,nodes)


def get_vertices(a):
    edges = []
    for i in range(a.shape[1]):
        for j in range(i, a.shape[0]):
            if a[i, j] == 1:
                edges.append((i, j))
                # edges.append((j, i)) #for two dir
    return edges


def dgl_graph(img, prep = bool):
    image = nib.load(img)
    # transform to canonical coordinates (RAS)
    image = nib.as_closest_canonical(image)
    image.set_data_dtype(np.uint8)
    image = image.get_fdata()
    image = image[200:205, 200:205, 100:105]
    if prep :
        image = normalize1(normalize(image*-1))
    adj = img2graph(image)[0]
    #node_list = img2graph(image)[1][0]
    node_features = (img2graph(image)[1]).long()
    edge_list = np.array(get_vertices(adj))
    #edge_features = torch.tensor(np.ones_like(edge_list))
    # Create the DGL Graph
    g = dgl.DGLGraph()
    g.add_nodes(node_features.size(0))
    g.ndata['feat'] = node_features.long()

    for src, dst in edge_list:
        g.add_edges(src.item(), dst.item())
    edge_feat_dim = 1
    g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim)
    #g.edata['feat'] = edge_features
    return g


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']


        This function is called inside a function in MoleculeDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float()

    return g


class NewDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs=None, cross_val=5):
        self.data_dir = data_dir
        self.split = split
        self.cross_val = cross_val
        '''with open(data_dir + "/%s.pickle" % self.split, "rb") as f:
            self.data = pickle.load(f)
        with open(data_dir + "/imagesTr", "rb") as l:
            self.data_label = pickle.load(l)'''
        self.data = create_lists_from_splitted_dataset(data_dir)
        self.graph_labels = []
        self.graph_lists = []
        self.n_samples = len(self.data)
        self.num_graphs = num_graphs
        self._split()

        self._prepare()

    def _split(self):  # TODO: ' Optimize cross validation for random case !!!'
        trunc = int(len(self.data) / self.cross_val)
        if self.split == 'val':
            self.data = self.data[0:trunc]
        else:
            self.data = self.data[trunc:len(self.data)]
        self.num_graphs = len(self.data)

    def _prepare(self):
        print("preparing %d graphs of %d for the %s set..." % (self.num_graphs, self.n_samples, self.split))
        i = 0
        for img, img_lab in self.data:
            i += 1
            self.graph_lists.append(dgl_graph(img, prep = True))
            self.graph_labels.append(dgl_graph(img_lab , prep = False).ndata['feat'])
            print('graph', i, 'of', self.num_graphs)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.num_graphs

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


class DatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='Task006_Lung'):
        t0 = time.time()
        self.name = name
        data_dir = 'C:/Users/localadmin/OneDrive/Maxime_Bollengier - Copy/nnUNet_raw_data_base/nnUNet_raw_data'
        if self.name == 'Task006_Lung':
            data_dir = data_dir + '/Task006_Lung'
            self.train = NewDatasetDGL(data_dir, 'train')
            self.val = NewDatasetDGL(data_dir, 'val')
            self._save(data_dir)
        elif self.name == 'Task007_Pancreas':
            data_dir = data_dir + '/Task006_Lung'
            self.train = NewDatasetDGL(data_dir, 'train')
            self.val = NewDatasetDGL(data_dir, 'val')
            self._save(data_dir)
        else:
            data_dir = data_dir + '/Task003_Liver'
            self.train = NewDatasetDGL(data_dir, 'train')
            self.val = NewDatasetDGL(data_dir, 'val')
            self._save(data_dir)
        print("Time taken: {:.4f}s".format(time.time() - t0))

    def _save(self, data_dir):
        start = time.time()
        with open(data_dir + '/'+self.name + '.pkl', 'wb') as f:
            pickle.dump([self.train, self.val], f)
        print(' Graph saved : Time (sec):', time.time() - start)


class MsdDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        start = time.time()
        print("[I] Loading dataset %s..." % name)
        self.name = name
        data_dir = 'C:/Users/localadmin/OneDrive/Maxime_Bollengier - Copy/nnUNet_raw_data_base/nnUNet_raw_data/' + name +'/'
        with open(data_dir + name + '.pkl', "rb") as f:
            f = pickle.load(f)

            self.train = self.create_artificial_features(f[0])
            self.val = self.create_artificial_features(f[1])

        print('train, val sizes :', len(self.train), len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # create artifical data feature (= in degree) for each node
    def create_artificial_features(self,dataset):
        for (graph, _) in dataset:
           ''' graph.ndata['feat'] = graph.in_degrees().view(-1, 1).int()
            graph.edata['feat'] = torch.ones(graph.number_of_edges(), 1)'''
        return dataset
    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        # tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        # tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        # snorm_n = torch.cat(tab_snorm_n).sqrt()
        # tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        # tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        # snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels

    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim=0)  # .squeeze()
        deg_inv = torch.where(deg > 0, 1. / torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))

    def _add_self_loops(self):

        # function for adding self loops
        # this function will be called only if self_loop flag is True

        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _add_positional_encodings(self, pos_enc_dim):

        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        # self.test.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]


def LoadData(DATASET_NAME): # TODO: add pre-processing mode
    """
        This function is called in the main.py file
        returns:
        ; dataset object
    """
    if DATASET_NAME == 'Task006_Lung':
        return MsdDataset(DATASET_NAME)

    if DATASET_NAME == 'Task007_Pancreas':
        return MsdDataset(DATASET_NAME)

    if DATASET_NAME == 'Task003_Liver':
        return MsdDataset(DATASET_NAME)


def PreprocessData(DATASET_NAME):
    """
            This function is called in the main.py file
            returns:
            ; dataset object
        """
    if DATASET_NAME == "Task006_Lung":
        return DatasetDGL(DATASET_NAME)

    if DATASET_NAME == 'Task007_Pancreas':
        return DatasetDGL(DATASET_NAME)

    if DATASET_NAME == 'Task003_Liver':
        return DatasetDGL(DATASET_NAME)


def main(Name):

    PreprocessData(Name)
    #dataset = LoadData(Name)
    #print(dataset)


if __name__ == "__main__":
    Name = 'Task006_Lung'
    main(Name)

'''
#dataset=PreprocessData('Task006_Lung')
dataset = LoadData('Task006_Lung')
q=dataset.train[0][1]
p=dataset.train[0][1]
print(q)
dataset = LoadData('Task006_Lung')
batch_size = 10
collate = MsdDataset.collate()
trainset = dataset.train
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate)
'''