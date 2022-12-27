import os
import sys
import shutil

sys.path.append('..')

from typing import List, Optional, Sequence, Union, Any, Callable

from torch_geometric.data import LightningDataset, InMemoryDataset, Data
from torch_geometric.loader import DataListLoader
from torch_geometric.transforms import Compose, RandomJitter, RandomRotate

from pytorch_lightning import LightningDataModule

from torch import load
from numpy import array, min
from scipy.sparse import coo_matrix



from utils.distances import *
from misc.utils import *
from misc.samplers import *

import requests
import tarfile

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
        for k, v in response.cookies.items():
            if k.startswith('download_warning'):
                return v
        return None
    
def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def download_FAUST2500_dataset(data_dir):
        
        if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
            print("Downloading FAUST2500 dataset...")

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




class Faust2500(InMemoryDataset):

    def __init__(self,
                root: str,
                train: bool = True,
                #test: bool = False,
                transform: Optional[Callable] = None):

                super().__init__(root=root, transform=transform)

                self.data, self.slices = load(self.processed_paths[0])

                # TODO
                # CONTROLLARE IL PERCHE' DI QUESTO CONTROLLO:
                if train:
                    self.data, self.slices = self.collate([self.get(i) for i in range(0,80)]) # DA RESETTARE A 80
                else:
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
            s.VERT[:,1] -= 0.85 + min(s.VERT[:,1])
            s.Adj = coo_matrix(calc_adj_matrix(s.VERT, s.TRIV))
            s.id = i//10
            shapes.append(s)
        
        shapes = array(shapes)

        # Pre-compute geodesic distance for each shape
        with torch.no_grad():
            for si in range(len(shapes)):
                #print("Processing shape %d" % si)
                V = torch.from_numpy(shapes[si].VERT).float().cuda()[None, ...]
                T = torch.from_numpy(shapes[si].TRIV).long().cuda()[None,...]
                D, _, _, _, _, _ = distance_GIH(V, T)
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
    
    
    

        
            


class VAEDataset(LightningDataModule):

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 4,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.data_path = data_path
        self.dataset = data_path.split("/")[-1]
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
    
    def setup(self, stage: Optional[str] = None) -> None:

        train_transform = Compose([
            RandomRotate(36, axis=1),
            RandomJitter(0.005)
        ])


        ### DATASETS ###

        if self.dataset.lower() == "faust-2500":

            if stage == "fit" or stage == None:
                self.train_dataset = Faust2500(self.data_path, train=True, transform=train_transform)
                #self.val_dataset = Faust2500(self.data_path, train=False, transform=train_transform)
        
    ### DATALOADERS ###

    def train_dataloader(self) -> List[DataListLoader]:
        intra_loader = DataListLoader(self.train_dataset, batch_size=2, num_workers=self.num_workers, sampler=IntraSampler(self.train_dataset))
        interp_loader = DataListLoader(self.train_dataset, batch_size=2, shuffle=True, num_workers=self.num_workers)
        rec_loader = DataListLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers)

        loaders = {"intra": intra_loader, "interp": interp_loader, "rec": rec_loader}
        return loaders
        #return [intra_loader, interp_loader, rec_loader]

    
    # TODO:
    # definire val_dataloaders
