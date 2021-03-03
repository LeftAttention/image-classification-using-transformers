# image-classification-using-transformers
pixel level attention paired with patch level attention for image classification

## Usage

```python
import torch
from transformer import TNT

tnt = TNT(
    image_size = 256,       # size of image
    patch_dim = 512,        # dimension of patch token
    pixel_dim = 24,         # dimension of pixel token
    patch_size = 16,        # patch size
    pixel_size = 4,         # pixel size
    depth = 6,              # depth
    num_classes = 1000,     # output number of classes
    attn_dropout = 0.1,     # attention dropout
    ff_dropout = 0.1        # feedforward dropout
)

img = torch.randn(2, 3, 256, 256)
logits = tnt(img) # (2, 1000)
```
