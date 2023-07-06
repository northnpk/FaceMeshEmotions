# Installation guide
## If you are using on PC or Mac
```
pip3 install -r requirements.txt
pip3 install mediapipe==0.10.1
```
## If you are using other devices
You should manually install the mediapipe
```
pip3 install -r requirements.txt
```
## Install Other required for graph models (GCN, GAT, etc.)
```
import torch
import os
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
```
```
!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q git+https://github.com/pyg-team/pyg-lib.git
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
```
