import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

def get_faim_writer(args):
    BASE_DIR = '/faim/tensorboard_runs'
    
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    try:
        if args.checkpoint == '':
            checkpoint = 'none'
        else:
            checkpoint = args.checkpoint
        dir_string = os.path.join(BASE_DIR, args.tensorboard_dir, f'm_{args.model}_d_{args.dataset}_e_{args.epochs}_b_{args.batch_size}_cp_{checkpoint}_{current_time}_{args.comment}')
    except:
        dir_string = os.path.join(BASE_DIR, f'unknown_{current_time}')
    print('Tensorboard dir_string: ', dir_string)
    return SummaryWriter(log_dir=dir_string)
        
