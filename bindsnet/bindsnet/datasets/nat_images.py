from typing import Optional, Tuple, List, Iterable, Dict
import torch
import numpy as np

from PIL import Image
from ..encoding import Encoder, NullEncoder

import sys
sys.path.append("C:/Users/cstei/Dropbox/work/projects/hebb/code/")
import libstein.hebb.image_data as image_data

class NatImages(torch.utils.data.Dataset):

    def __init__(
        self,
        data: Optional[np.array] = None,
        image_encoder: Optional[Encoder] = None,
        label_encoder: Optional[Encoder] = None,
        transform = None,
        *args,
        **kwargs
    ):
        # language=rst
        """
        :param image_encoder: Spike encoder for use on the image
        :param label_encoder: Spike encoder for use on the label
        :param *args: Arguments for the original dataset
        :param **kwargs: Keyword arguments for the original dataset
        """
        super().__init__(*args, **kwargs)

        self.args = args
        self.kwargs = kwargs

        # Allow the passthrough of None, but change to NullEncoder
        if image_encoder is None:
            image_encoder = NullEncoder()

        if label_encoder is None:
            label_encoder = NullEncoder()

        if data:
        	self.data = data
        else:
        	self.data = image_data.oef_patches(ndata = 100000, patsize = 28)

        self.image_encoder = image_encoder
        self.label_encoder = label_encoder

        self.transform = transform

    def __getitem__(self, ind: int) -> Dict[str, torch.Tensor]:
        image = Image.fromarray(self.data[ind])
        label = 0

        if self.transform:
            image = self.transform(image)

        output = {
            "image": image,
            "label": label,
            "encoded_image": self.image_encoder(image),
            "encoded_label": self.label_encoder(label),
        }

        return output

    def __len__(self):
        return self.data.shape[0]
