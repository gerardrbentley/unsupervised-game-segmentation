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
            checkpoint = str(checkpoint).replace(os.sep, '_')
        dir_string = os.path.join(BASE_DIR, args.tensorboard_dir, f'm_{args.model}_d_{args.dataset}_e_{args.epochs}_b_{args.batch_size}_cp_{checkpoint}_{current_time}_{args.comment}')
        print_string = f'Started Training for {args.model} (initialized at checkpoint: {checkpoint}) on dataset {args.dataset} for {args.epochs} epochs, batch size {args.batch_size}. Comment: {args.comment}. DT: {current_time}'
    except:
        dir_string = os.path.join(BASE_DIR, f'unknown_{current_time}')
        print_string = f'Could not Parse Project Args'
    print('Tensorboard dir_string: ', dir_string)
    the_writer = SummaryWriter(log_dir=dir_string)
    the_writer.add_text('Init_Text', print_string)
    return the_writer
        
