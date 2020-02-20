# Game Texture Segmentation

Experiments and research in segmenting image textures from pixelated game screenshots

## Install
If working with [conda](https://docs.conda.io/en/latest/miniconda.html) you can use the following to set up a virtual python environment.
```
conda create --name mldev python=3.8
conda activate mldev
```
Then you can use pip install to get all the dependencies (this works with virtualenv as well)
```
pip install -r requirements.txt
```

The pydensecrf library had issues installing on my system via `pip install pydensecrf`, so I excluded from requirements. If `pip install pydensecrf` throws errors try using  `pip install git+https://github.com/lucasb-eyer/pydensecrf.git`

## Run Pre-trained test

The following will load the model checkpoint at `saved_models/mario_overfit.pth`, run `overfit.png` through the model, and display the resulting images and mask
```
python train.py --checkpoint 'saved_models/mario_overfit.pth' --test-only --visualize
```
To make sure you get your flags right you can also run the Model without training with the `evaluation.py` script (in which case the `--test-only` flag is redundant but `--visualize` still dictates if matplot lib shows a pop up visual)
```
python evaluation.py -cp saved_models/mario_overfit.pth --visualize
```

## Overfit Training

A good practice of testing a new model is getting it to Overfit a sample dataset. In our case we want one image to be encoded, decoded, and segmented extremely well.

In `datasets.py` is an `OverfitDataset` that defaults to using the image `overfit.png` 2000 times per epoch (and 1 time for validation / evaluation loop).

Recommended transforms for this model: 
```
    import input_target_transforms as TT
    transforms = []

    # Input size of model is 224x224
    transforms.append(TT.Resize(224))
    # PIL Images to Torch Tensors
    transforms.append(TT.ToTensor())
    # Normalize Images for better Gradient Descent
    transforms.append(TT.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    transforms = TT.Compose(transforms)
```
** TODO ** Check the mean and std on our dataset, these are the values to Normalize over ImageNet (Or some other natural image datasets I believe)

Running the command `python train.py --dataset overfit --epochs 5 --no-aug` should start the training on your machine. `python train.py --help` will show all available Command Line Flags (or look at the `parse_args()` function of `ml_args.py`). You may need to lower --batch-size and --workers depending on your computer's computing abilities.

## Distributed Training

Training on multiple GPUS is simple using pytorch's distributed launch.py utility

`python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --use_env train.py --relevant --training --flags`

## Resources

W Net Pytorch Implementation for Unsupervised image Segmentation: https://github.com/fkodom/wnet-unsupervised-image-segmentation

Dense Conditional Random Field library: https://github.com/lucasb-eyer/pydensecrf

Pytorch Training Utils and Distributed Utils references: https://github.com/pytorch/vision/tree/master/references/segmentation


## Papers
```
W-Net: A Deep Model for Fully Unsupervised Image Segmentation
Xide Xia and Brian Kulis
2017
```

```
Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
Philipp Krähenbühl and Vladlen Koltun
NIPS 2011
```
