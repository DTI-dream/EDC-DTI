import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import random


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir)
  
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()

def setup_seed(seed=12345):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True       
