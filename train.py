import datetime
import os
import time

import torch
from torch import nn
import torchvision

import input_target_transforms as TT
import distributed_utils

from ml_args import parse_args
from models import WNet
from loss import NCutLoss2D, OpeningLoss2D
from visualize import visualize_outputs, matplotlib_imshow, argmax_to_rgb
from crf import crf_batch_fit_predict

from datasets import GameImagesDataset, GameFoldersDataset, OverfitDataset

# Reference Training Script and Utils: https://github.com/pytorch/vision/tree/master/references


def get_dataset(name, train_or_val, transform):
    paths = {
        "overfit": ('./overfit.png', OverfitDataset),
        "test_mario": ('/faim/datasets/test_mario', GameImagesDataset),
        "mario": ('/faim/datasets/mario_images', GameImagesDataset),
        "blap": ('/faim/datasets/blap_images', GameFoldersDataset)
    }
    p, ds_fn = paths[name]

    ds = ds_fn(root=p, train_or_val=train_or_val, transform=transform)
    return ds



# TODO verify these normalizations work well for pixelated games
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
def get_transform(train):
    base_size = 224
    crop_size = 180

    transforms = []
    if train:
        # min_size = int(0.5 * base_size)
        max_size = int(2.5 * base_size)

        transforms.append(TT.RandomResize(base_size, max_size))
        
        transforms.append(TT.RandomHorizontalFlip(0.5))
        transforms.append(TT.RandomVerticalFlip(0.5))

        transforms.append(TT.RandomCrop(crop_size))

    transforms.append(TT.Resize(base_size))
    transforms.append(TT.ToTensor())
    transforms.append(TT.Normalize(mean=DEFAULT_MEAN,
                                  std=DEFAULT_STD))

    return TT.Compose(transforms)

# TODO verify these weights from W-Net implementation
# Weights for NCutLoss2D, MSELoss, and OpeningLoss2D, respectively
ALPHA, BETA, GAMMA = 1e-3, 1, 1e-1
def criterion(inputs, target):
    result_masks, result_reconstructions = inputs["mask"], inputs["reconstruction"]
    
    result_reconstructions = result_reconstructions.contiguous()
    target = target.contiguous()

    
    soft_cut_loss = ALPHA * NCutLoss2D()(result_masks, target)
    reconstr_loss = BETA * nn.MSELoss()(result_reconstructions, target.detach())
    smooth_loss = GAMMA * OpeningLoss2D()(result_masks)
    total_loss = soft_cut_loss + reconstr_loss + smooth_loss
    
    return total_loss, soft_cut_loss, reconstr_loss, smooth_loss

# TODO put in it's own file
def evaluate(model, data_loader, device, epoch=0, writer=None):
    model.eval()
    metric_logger = distributed_utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        for data, i in metric_logger.log_every(data_loader, 20, header):
        
            step = epoch * len(data_loader.dataset) + i

            image, target = data['image'], data['target']
            image, target = image.to(device), target.to(device)
            output = model(image)
            total_loss, soft_cut_loss, reconstr_loss, smooth_loss = criterion(output, target)

            loss_item = total_loss.item()
            metric_logger.update(total_loss=loss_item, soft_cut_loss=soft_cut_loss, reconstr_loss=reconstr_loss, smooth_loss=smooth_loss)

            if writer is not None:
                writer.add_scalar('Loss_Sum/Validation', loss_item, global_step=step)
                writer.add_scalar('Loss_Soft_Cut/Validation', soft_cut_loss.item(), global_step=step)
                writer.add_scalar('Loss_Reconstruction/Validation', reconstr_loss.item(), global_step=step)
                writer.add_scalar('Loss_Smoothing/Validation', smooth_loss.item(), global_step=step)

                # TODO: write function to visualize mask and output in eval loop
                if i < 5:
                    mask, reconstruction = output['mask'], output['reconstruction'].squeeze(0)
                    np_mask = mask.detach().cpu().numpy()
                    np_image = image.detach().cpu().numpy()
                    np_new_mask = crf_batch_fit_predict(np_mask, np_image)

                    mask_viz = mask.to(device).argmax(1).float()
                    new_mask_viz = torch.from_numpy(np_new_mask).to(device).argmax(1).float()
                    
                    mask_viz = argmax_to_rgb(mask_viz).unsqueeze(0)
                    new_mask_viz = argmax_to_rgb(new_mask_viz).unsqueeze(0)

                    input_images = [image, target]
                    result_images = [mask_viz, new_mask_viz]
                    visualize_images = [image, target, reconstruction, mask_viz, new_mask_viz]

                    # for img in visualize_images:
                    #     print('--------')
                    #     print(type(img), img.shape, img.dtype)
                    #     print(img.min().item(), img.max().item())
                        # print(img.unique())
                    
                    img_grid = torchvision.utils.make_grid(torch.cat(input_images), nrow=2, normalize=True)
                    writer.add_image(f'Validation_Sample_{i}/Input_And_Target', img_grid, global_step=step)

                    reconstruction = TT.img_norm(reconstruction)
                    writer.add_image(f'Validation_Sample_{i}/AE_Reconstruction', reconstruction, global_step=step)

                    result_grid = torchvision.utils.make_grid(torch.cat(result_images), nrow=2, normalize=True)
                    writer.add_image(f'Validation_Sample_{i}/Raw_Mask_And_CRF_Mask', result_grid, global_step=step)
    return metric_logger


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, writer=None):
    model.train()
    metric_logger = distributed_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', distributed_utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data, i in metric_logger.log_every(data_loader, print_freq, header):
        step = epoch * len(data_loader.dataset) + i

        image, target = data['image'], data['target']
        image, target = image.to(device), target.to(device)
        output = model(image)
        total_loss, soft_cut_loss, reconstr_loss, smooth_loss = criterion(output, target)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        lr_scheduler.step()
        metric_logger.update(total_loss=total_loss.item(), soft_cut_loss=soft_cut_loss, reconstr_loss=reconstr_loss, smooth_loss=smooth_loss, lr=optimizer.param_groups[0]["lr"])
        if writer is not None and i % print_freq == 0:
            writer.add_scalar('Loss_Sum/Training', total_loss.item(), global_step=step)
            writer.add_scalar('Loss_Soft_Cut/Training', soft_cut_loss.item(), global_step=step)
            writer.add_scalar('Loss_Reconstruction/Training', reconstr_loss.item(), global_step=step)
            writer.add_scalar('Loss_Smoothing/Training', smooth_loss.item(), global_step=step)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]["lr"], global_step=step)

# TODO get rid of this or make it useful for eval loop logging to tensorboard
def visualize(model, dataset, device, writer=None):
    print(r'---------------------- VISUALIZE OUTPUTS ----------------------')
    idx = 0
    # input_image = dataset[idx]['image'].unsqueeze(0)
    # targ = dataset[idx]['target'].detach().cpu().numpy()
    data = dataset[idx]
    input_image = data['image'].unsqueeze(0)
    targ = data['target'].unsqueeze(0).detach().cpu().numpy()
    input_image.to(device)

    result = model(input_image)
    mask, reconstruction= result['mask'], result['reconstruction']
    input_image = input_image.detach().cpu().numpy()
    reconstruction = reconstruction.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    new_mask = crf_batch_fit_predict(mask, input_image)
    print(mask.shape)
    print(new_mask.shape)
    print(input_image.shape)
    print(targ.shape)
    visualize_outputs(input_image, targ, reconstruction, mask.argmax(1), new_mask.argmax(1),
                      titles=['Image', 'Target', 'AE Output', 'Raw Mask', 'CRF Mask'])

def main(args):
    if args.output_dir:
        distributed_utils.mkdir(args.output_dir)
    # Setup for Distributed if Available, Else set args.distributed to False
    distributed_utils.init_distributed_mode(args)   
    print(args)
    # Use device from args. Locally CPU, with GPU 'cuda', with Distributed 'cuda:x' where x is gpu number
    device = torch.device(args.device)

    # train=True applies augmentations to inputs such as flips and crops
    if args.no_augmentation:
        dataset = get_dataset(args.dataset, "train", get_transform(train=False))
    else:
        dataset = get_dataset(args.dataset, "train", get_transform(train=True))
    dataset_test = get_dataset(args.dataset, "val", get_transform(train=False))
    print(f'len train set: {len(dataset)} ; len test set: {len(dataset_test)}')

    # Distributed mode chunks the dataset so that each worker does equal work but doesn't do extra work
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # Configured to fetch the correct batched data
    # Pin Memory should help with shared CUDA data resources
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=True, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        pin_memory=True)
    
    # Initialize Model, handling distributed as needed
    model = WNet()
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Fetch Model weights from a checkpoint. Checkpoint saved in distributed_utils.py
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    # For analyzing model parameters and saving the master weights
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Don't log to tensorboard if flag not to or distributed training and not master thread
    if args.no_tensorboard or ('rank' in args and args.rank != 0):
        writer = None
    else:
        from faim_tensorboard import get_faim_writer
        writer = get_faim_writer(args)

        # Add a training image and it's target to tensorboard
        rand_select = torch.randint(0, len(dataset), (6,)).tolist()
        train_images = []
        for idx in rand_select:
            data = dataset[idx]
            image, target = data['image'], data['target']
            
            train_images.append(image)
            train_images.append(target)
        
        img_grid = torchvision.utils.make_grid(train_images, nrow=6, normalize=True)
        writer.add_image('Random_Train_Sample', img_grid)

        # TODO look into adding graph to tensorboard. Might need easier example
        # writer.add_graph(model_without_ddp, image.unsqueeze(0))
  
    # TODO Put this in another file
    if args.test_only:
        eval_result = evaluate(model, data_loader_test, device,epoch=0, writer=writer)
        print(eval_result)
        if args.do_visualize:
            visualize(model, dataset_test, device, writer)
        return

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.parameters() if p.requires_grad]},
    ]

    if args.distributed:
        args.lr = args.lr * args.world_size

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, writer)
        if data_loader_test is not None:
            result_metric_logger = evaluate(model, data_loader_test, device, epoch, writer)
            # print(result_metric_logger)
        else:
            result_metric_logger = None

        distributed_utils.save_on_master(
            {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args
            },
            os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if writer is not None:
        world = args.world_size if 'rank' in args else 0

        param_dict = {
            'hp/epochs': args.epochs,
            'hp/num_samples': len(dataset),
            'hp/batch_size': args.batch_size,
            'hp/lr_start': args.lr,
            'hp/momentum': args.momentum,
            'hp/weight_decay': args.weight_decay,
            'hp/distributed': int(args.distributed),
            'hp/world_size': world,
        }
        if result_metric_logger is not None:
            result_dict = {
                'res/total_loss' : getattr(result_metric_logger, "total_loss").value,
                'res/soft_cut_loss' : getattr(result_metric_logger, "soft_cut_loss").value, 
                'res/reconstr_loss' : getattr(result_metric_logger, "reconstr_loss").value, 
                'res/smooth_loss' : getattr(result_metric_logger, "smooth_loss").value,
            }
        else:
            result_dict = {}

        writer.add_hparams(param_dict, result_dict)
        writer.add_text('Training_Ended', f'Total Time to Train: {total_time_str}')
        writer.close()

    if args.do_visualize:
        visualize(model, dataset_test, device)

if __name__ == "__main__":
    args = parse_args()

    main(args)
