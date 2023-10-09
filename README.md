[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Vision Transformers for Facial Recognition (WIP)
An attempt to create the most accurate, reliable, and general vision transformers for facial recognition at scale.


## Installation
`pip install frvit`


## Usage
```python
import torch
from frvit.navit import NaViT

v = NaViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1,
    token_dropout_prob=0.1,  # token dropout of 10% (keep 90% of tokens)
)

# 5 images of different resolutions - List[List[Tensor]]

# for now, you'll have to correctly place images in same batch element as to not exceed maximum allowed sequence length for self-attention w/ masking

images = [
    [torch.randn(3, 256, 256), torch.randn(3, 128, 128)],
    [torch.randn(3, 128, 256), torch.randn(3, 256, 128)],
    [torch.randn(3, 64, 256)],
]

preds = v(images)  # (5, 1000) - 5, because 5 images of different resolution above



```


## Dataset strategy
Here is a table of some popular open source facial recognition datasets with metadata and source links:

| Dataset | Images | Identities | Format | Task | License | Source |
|-|-|-|-|-|-|-|  
| Labeled Faces in the Wild (LFW) | 13,233 | 5,749 | JPEG | Face verification | Creative Commons BY 4.0 | http://vis-www.cs.umass.edu/lfw/ |
| YouTube Faces (YTF) | 3,425 | 1,595 | JPEG | Face verification | Creative Commons BY 4.0 | https://www.cs.tau.ac.il/~wolf/ytfaces/ |
| MegaFace | 1 million | 690,572 | JPEG | Face identification | Creative Commons BY 4.0 | http://megaface.cs.washington.edu/ |  
| MS-Celeb-1M  | 10 million | 100,000 | JPEG | Face identification | Custom | https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/ |
| CASIA WebFace | 494,414 | 10,575 | JPEG | Face verification | Custom | http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html |
| FaceScrub | 107,818 | 530 | JPEG | Face identification | Custom | http://vintage.winklerbros.net/facescrub.html |
| VGG Face2 | 3.31 million | 9,131 | JPEG | Face verification, identification | Creative Commons BY 4.0 | https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/ |
| UMD Faces | 8,501 | 3,692 | JPEG | Face identification | Custom | https://www.umdfaces.io/ |
| CelebA | 202,599 | 10,177 | JPEG | Face attribute analysis | Creative Commons BY 4.0 | http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html |

# License
MIT

