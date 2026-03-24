# DBINet, IEEE GRSL 2024</a> </p>

- Paper: [Dual Backbone Interaction Network for Burned Area Segmentation in Optical Remote Sensing Images](https://ieeexplore.ieee.org/document/10445196)


## Abstract

The existing methods for burned area segmentation (BAS) in optical remote sensing images (ORSIs) mainly adopt convolution neural network (CNN) as the backbone, which has limited receptive filed and suffers from long-term dependencies problem. To address this issue, we propose a novel salient object detection (SOD) network (DBINet) based on dual interactive convolution-transformer backbone for BAS in ORSIs. DBINet combines the benefits of CNN and transformer: CNN is good at extracting local spatial information, while transformer does well in modeling long-term dependencies. The core component of DBINet is three newly designed modules: dual-feature fusion module (DFM), convertor, and a novel decoder. Specifically, DFM is proposed to bridge two different backbones. Convertor is designed to fuse the multiscale coarse features from the main encoder and produce the fined feature for the decoder. The decoder has a multilevel feature aggregating process and a self-refining process, which restores the resolution and generates the prediction results. Experiments on three datasets demonstrate that our DBINet outperforms the state-of-the-art methods and achieves the best S-measure on three datasets: 0.847, 0.888, and 0.883. Code is available at: https://github.com/Voruarn/DBINet.


## Related Works
[Burned Area Segmentation in Optical Remote Sensing Images Driven by U-Shaped Multistage Masked Autoencoder ](https://github.com/Voruarn/DCNet), IEEE JSTARS 2024.

[Controllable diffusion generated dataset and hybrid CNN–Mamba network for burned area segmentation ](https://github.com/Voruarn/HCM), ADVEI 2025.

[A novel salient object detection network for burned area segmentation in high-resolution remote sensing images ](https://github.com/Voruarn/PANet), ENVSOFT 2025.

```
## 📎 Citation

If you find the code helpful in your research or work, please cite the following paper(s).

@article{10445196,
  author={Fang, Wei and Fu, Yuxiang and Sheng, Victor S.},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Dual Backbone Interaction Network for Burned Area Segmentation in Optical Remote Sensing Images}, 
  year={2024},
  volume={21},
  number={},
  pages={1-5},
  }
```
