import os
import shutil

import numpy as np
import torch
import torch_geometric
import torch_geometric.data
from torch_geometric.data import Data, InMemoryDataset

import torch_geometric.io as gio
import torch_geometric.transforms as transforms
import tqdm

import requests
import tarfile

import scipy.sparse as sp

from utils.distances import *
from misc.utils import *

class Faust2500Dataset(InMemoryDataset):

    def __init__(self,
        root: str,
        device: torch.device=torch.device("cpu"),
        train: bool=True,
        test: bool=False,
        transform_data: bool=True):
        
        transform =  transforms.Compose([transforms.RandomRotate(36, axis=1),
                                        transforms.RandomTranslate(0.005)])
        
        super().__init__(root=root, transform=transform)

        self.data, self.slices = torch.load(self.processed_paths[0])


        if train and not test:
            self.data, self.slices = self.collate([self.get(i) for i in range(0,80)])
        elif test and not train:
            self.data, self.slices = self.collate([self.get(i) for i in range(80,100)])
        
        print(self.data)

        self.class_ids = [int(c) for c in self.data.y]
    
    def get(self, idx):
        data = super().get(idx)
        data.oripos = data.pos
        data.idx = idx
        return data
    
    @property
    def raw_file_names(self):
        return ["tr_reg_%03d_2000" % fi for fi in range(100)]
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        download_FAUST2500_dataset(self.raw_dir)
    
    def process(self):
        raw_names = self.raw_file_names
        shapes = []
        for i in range(len(raw_names)):
            s = lambda x : x
            [s.VERT, s.TRIV] = load_mesh(self.raw_dir + '/%s' % raw_names[i])
            s.VERT[:,1] -= 0.85 + np.min(s.VERT[:,1])
            s.Adj = sp.coo_matrix(calc_adj_matrix(s.VERT, s.TRIV))
            s.id = i//10
            shapes.append(s)
        
        shapes = np.array(shapes)
        NUM_POINTS = s.VERT.shape[0]

        # Pre-compute geodesic distance for each shape
        with torch.no_grad():
            for si in range(len(shapes)):
                #print("Processing shape %d" % si)
                V = torch.from_numpy(shapes[si].VERT).float().cuda()[None, ...]
                T = torch.from_numpy(shapes[si].TRIV).long().cuda()[None,...]
                D, grad, div, W, S, C = distance_GIH(V, T)
                shapes[si].Dg = D.data.float().cpu()
        
        # Convert to Data
        data_list = []
        for si in range(len(shapes)):
            data = Data(pos = torch.from_numpy(shapes[si].VERT).float(), y=shapes[si].id)
            data.faces = torch.from_numpy(shapes[si].TRIV).long()
            data.Dg = shapes[si].Dg

            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save( (data,  slices), self.processed_paths[0])

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params = {'id' : id}, stream = True)
    token =  get_confirm_token(response)

    if token:
        params = {'id' : id, 'confirm' : token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('dowload_warning'):
            return value
    
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def download_FAUST2500_dataset(data_dir):
    print("Downloading FAUST2500 dataset...")

    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        os.makedirs(data_dir, exist_ok=True)
        download_file_from_google_drive('18E4yYg2I0EJJ0wbTr1VCMWn-VIPmTwZW', data_dir + '/dataset.tar')

        tf = tarfile.open(data_dir + '/dataset.tar')
        tf.extractall(data_dir)
        tf.close()

        source_folder = data_dir + '/FAUST/'
        target_folder = data_dir 

        for file_name in os.listdir(source_folder):
            shutil.move(source_folder + file_name, target_folder + '/' + file_name)

        shutil.rmtree(source_folder)
        #os.remove(data_dir + '/dataset.tar')
        os.system('del %s' % (data_dir+'\dataset.tar'))