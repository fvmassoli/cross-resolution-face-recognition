# Cross-resolution learning for Face Recognition

This repository contains the code relative to the paper "[Cross-resolution learning for Face Recognition](https://www.sciencedirect.com/science/article/pii/S0262885620300597)" by Fabio Valerio Massoli (ISTI - CNR), Giuseppe Amato (ISTI - CNR), and Fabrizio Falchi (ISTI - CNR).

It reports a new training procedure for cross-resolution robust deep neural network. 

**Please note:** 
We are researchers, not a software company, and have no personnel devoted to documenting and maintaing this research code. Therefore this code is offered "AS IS". Exact reproduction of the numbers in the paper depends on exact reproduction of many factors, including the version of all software dependencies and the choice of underlying hardware (GPU model, etc). Therefore you should expect to need to re-tune your hyperparameters slightly for your new setup.

## Cross-resolution training

Proposed training approach

<p align="center">
<img src="https://github.com/fvmassoli/cross-resolution-face-recognition/blob/master/images/paper_training_algorithm.png"  alt="t-SNE" width="600" height="300">
</p>


2D t-SNE embeddings for 20 different identities randomly extracted from the VGGFace2 dataset. All the images were down-sampled to a resolution of 8 pixels. Left: “Base Model”. Right: model trained with our approach.

<p align="center">
<img src="https://github.com/fvmassoli/cross-resolution-face-recognition/blob/master/images/vggface_tsne_base_ft_models_8.png" alt="t-SNE" width="700" height="300">
</p>

## How to run the code
The current version of the code requires python 3.6 and pytorch 1.4.0.

Inside the dataset folder, the code expects to find two subdirs: "train" and "validation".

Minimal usage:

```
python -W ignore main.py --model-base-path path_to_base_model_weight_file --dset-base-path path_to_data_folder 
```

The base model is the SE-ResNet-50 (pretrained on the VGGFace2 dataset) that is available [here](https://github.com/fvmassoli/cross-resolution-face-recognition/releases/tag/v1.0).

The model is the SE-ResNet-50 with features dim = 2048.

**BE VERY CAREFUL**

When you download the VGGFace2 dataset, you should NOT use the test set while training. To create a validation set, just take a subset of the training set

## Reference
For all the details about the training procedure and the experimental results, please have a look at the [paper](https://www.sciencedirect.com/science/article/pii/S0262885620300597).

To cite our work, please use the following form

```
@article{massoli2020cross,
  title={Cross-resolution learning for Face Recognition},
  author={Massoli, Fabio Valerio and Amato, Giuseppe and Falchi, Fabrizio},
  journal={Image and Vision Computing},
  pages={103927},
  year={2020},
  publisher={Elsevier}
}
```

## Contacts & Model Request
If you have any question about our work, please contact [Dr. Fabio Valerio Massoli](mailto:fabio.massoli@isti.cnr.it). 

**Currently, we cannot supply the trained model checkpoint.**

Have fun! :-D
