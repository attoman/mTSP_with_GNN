# utils/seed.py

import torch
import random
import numpy as np

def set_seed(seed):
    """
    재현성을 위해 랜덤 시드를 설정합니다.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
