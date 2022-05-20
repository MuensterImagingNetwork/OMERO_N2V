# OMERO

### Convenience Functions:
Collection of functions related to the OMERO Image database for easy integration in Python scripts.

Code was adapted from the offical OMERO guide (https://omero-guides.readthedocs.io/en/latest/python/docs/gettingstarted.html), OMERO Python language bindings documentation (https://docs.openmicroscopy.org/omero/5.6.0/developers/Python.htm) and the ilastik API - OMERO tutorial (https://omero-guides.readthedocs.io/en/latest/ilastik/docs/gettingstarted.html).  



# N2V

Interactive GUI to train and apply the DeepLearningTool Noise2Void for image denoising. 

The GUI implements the original N2V code by Krull et al., Noise2Void - Learning Denoising from Single Noisy Images (https://arxiv.org/abs/1811.10980)
Source code: https://github.com/juglab/N2V_fiji/

## Installation

```shell
conda create -n n2v python=3.7
conda activate n2v
```

With GPU
```shell
conda install tensorflow-gpu=2.4.1 keras=2.3.1
pip install n2v omero-py
```

Aleternative: With CPU only
```shell
pip install tensorflow==2.4 keras=2.3.1
pip install n2v omero-py
```

More details on the installation of N2V in [here](https://github.com/juglab/n2v)
