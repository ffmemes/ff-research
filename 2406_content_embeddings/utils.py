from typing import List
from os import listdir
from os.path import join
import torch
import torch.nn.functional as F
 
from PIL import Image


class Similars:
    def __init__(self, embeddings: torch.Tensor) -> None:
        self.embeddings = embeddings
    
    def similar(self, id: int, k: int = 5) -> List[int]:
        similarity = F.cosine_similarity(self.embeddings[id], self.embeddings)

        _, top_k_indices = torch.topk(similarity, k=k)
        # [1:] - to remove the item itself
        top_k_indices = top_k_indices[1:]
        return [int(el) for el in list(top_k_indices.numpy())]


def get_images(path='./images/general_ru', batch_size=32):
    fns = listdir(path)
    fns = [fn for fn in fns if fn[-3:] == 'png']
    idx = 0
    while idx * batch_size < len(fns):
        sl = slice(idx*batch_size,(idx+1)*batch_size)
        yield fns[sl], [Image.open(join(path, fn)) for fn in fns[sl]]
        idx += 1