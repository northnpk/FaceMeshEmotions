```
pip3 install -r requirements.txt
```
```
import torch
import os
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
```
```
!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q git+https://github.com/pyg-team/pyg-lib.git
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
```
