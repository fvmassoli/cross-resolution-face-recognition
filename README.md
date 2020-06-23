# Cross-resolution learning for Face Recognition

This repository contains the code relative to the paper "[Cross-resolution learning for Face Recognition](https://www.sciencedirect.com/science/article/pii/S0262885620300597)" by Fabio Valerio Massoli (ISTI - CNR), Giuseppe Amato (ISTI - CNR), and Fabrizio Falchi (ISTI - CNR).

It reports a new training procedure for cross-resolution robust deep neural network. 

**(WIP ... all the code will be soon available)**

## Cross-resolution training

Proposed training approach

<p align="center">
<img src="https://github.com/fvmassoli/cross-resolution-face-recognition/blob/master/images/paper_training_algorithm.png"  alt="t-SNE" width="600" height="300">
</p>


2D t-SNE embeddings for 20 different identities randomly extracted from the VGGFace2 dataset. All the images were down-sampled to a resolution of 8 pixels. Left: “Base Model”. Right: model trained with our approach.

<p align="center">
<img src="https://github.com/fvmassoli/cross-resolution-face-recognition/blob/master/images/vggface_tsne_base_ft_models_8.png" alt="t-SNE" width="700" height="300">
</p>


## Reference
For all the details about the training procedure and the experimental results, please have a look at the [paper](https://www.sciencedirect.com/science/article/pii/S0262885620300597).

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

## Contacts
If you have any question about our work, please contact [Dr. Fabio Valerio Massoli](mailto:fabio.massoli@isti.cnr.it). 

We make our cross-resolution best model available to researchers upon request to: [Dr. Fabio Valerio Massoli](mailto:fabio.massoli@isti.cnr.it).


Have fun! :-D
