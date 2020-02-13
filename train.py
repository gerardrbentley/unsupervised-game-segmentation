import datetime
import os
import time

import torch
from torch import nn

import input_target_transforms as TT
import distributed_utils

from models import WNet
from loss import NCutLoss2D, OpeningLoss2D
from visualize import visualize_outputs
from crf import crf_batch_fit_predict

from datasets import GameImagesDataset, OverfitDataset

# Reference Training Script and Utils: https://github.com/pytorch/vision/tree/master/references

def get_dataset(name, train_or_val, transform):
    paths = {
        "overfit": ('./overfit.png', OverfitDataset),
        "test_mario": ('/faim/datasets/test_mario', GameImagesDataset),
        "mario": ('/faim/datasets/mario_images', GameImagesDataset)
    }
    p, ds_fn = paths[name]

    ds = ds_fn(root=p, train_or_val=train_or_val, transform=transform)
    return ds


def get_transform(train):
    base_size = 224
    crop_size = 180

    transforms = []
    if train:
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        transforms.append(TT.RandomResize(min_size, max_size))
        
        transforms.append(TT.RandomHorizontalFlip(0.5))
        transforms.append(TT.RandomVerticalFlip(0.5))

        transforms.append(TT.RandomCrop(crop_size))

    transforms.append(TT.Resize(base_size))
    transforms.append(TT.ToTensor())
    transforms.append(TT.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return TT.Compose(transforms)


def criterion(inputs, target):
    result_masks, result_reconstructions = inputs["mask"], inputs["reconstruction"]
    
    result_reconstructions = result_reconstructions.contiguous()
    target = target.contiguous()

    # Weights for NCutLoss2D, MSELoss, and OpeningLoss2D, respectively
    alpha, beta, gamma = 1e-3, 1, 1e-1
    ncut_loss = alpha * NCutLoss2D()(result_masks, target)
    mse_loss = beta * nn.MSELoss()(result_reconstructions, target.detach())
    smooth_loss = gamma * OpeningLoss2D()(result_masks)
    loss = ncut_loss + mse_loss + smooth_loss
    
    return loss

    # for name, x in inputs.items():
    #     losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    # if len(losses) == 1:
    #     return losses['out']

    # return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device):
    model.eval()
    eval_result = distributed_utils.SmoothedValue()
    metric_logger = distributed_utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', distributed_utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Test:'
    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 20, header):
            image, target = data['image'], data['target']
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)

            loss_item = loss.item()
            metric_logger.update(loss=loss_item)
            eval_result.update(loss_item)
        eval_result.synchronize_between_processes()
    return eval_result


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = distributed_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', distributed_utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for data in metric_logger.log_every(data_loader, print_freq, header):
        image, target = data['image'], data['target']
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

def visualize(model, dataset, device):
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

    distributed_utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.no_augmentation:
        dataset = get_dataset(args.dataset, "train", get_transform(train=False))
    else:
        dataset = get_dataset(args.dataset, "train", get_transform(train=True))
    dataset_test = get_dataset(args.dataset, "val", get_transform(train=False))
    print(f'len train set: {len(dataset)} ; len test set: {len(dataset_test)}')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        )

    model = WNet()
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.test_only:
        eval_result = evaluate(model, data_loader_test, device=device)
        print(eval_result)
        if args.do_visualize:
            visualize(model, dataset_test, device)
        return

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.parameters() if p.requires_grad]},
    ]
    # if args.aux_loss:
    #     params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
    #     params_to_optimize.append({"params": params, "lr": args.lr * 10})
    
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
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq)
        if data_loader_test is not None:
            eval_result = evaluate(model, data_loader_test, device=device)
            print(eval_result)

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
    if args.do_visualize:
        visualize(model, dataset_test, device)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training')

    parser.add_argument('--dataset', default='overfit', help='dataset')
    parser.add_argument('--model', default='wnet', help='model')
    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('-e', '--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-o', '--output-dir', default='./output/', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--no-aug",
        dest="no_augmentation",
        help="Don't augment training images",
        action="store_true",
    )
    parser.add_argument(
        "--visualize",
        dest="do_visualize",
        help="Visualize the model outputs after train / test",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
