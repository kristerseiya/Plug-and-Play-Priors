# What is Plug-and-Play Priors?
Plug-and-Play is a way of making the optimization of image restoration problem easier by decoupling the data fidelity term (forward model) and regularization term (image prior), thus allows you to plug in arbitrary gaussian denoisers for different image restoration tasks.

This is an attempted implementation of Plug-and-Play ADMM.

## Image Inpainting
![Alt text](result/tiger_dncnn.png?raw=true "Image Inapinting")

```bash
python inpaint.py \
       --image imgs/tiger_224x224.png \
       --sample 0.2 \
       --prior dncnn \
       --alpha 2000 \
       --iter 100 \
       --weights Denoisers/dnsr/DnCNN/weights/dncnn50_17.pth \
       --verbose
```
```markdown
python inpaint.py -h

  --image IMAGE               path to image
  --idx IDX                   File with index of sampled points (binary int32)
  --sample SAMPLE             sample rate from 0 to 1 (default 0.2)
  --noise NOISE               add gaussian noise with given noise level (default 0)
  --prior PRIOR               image prior option ['dct' or 'dncnn' or 'tv' or 'bm3d']
  --iter ITER                 number of iteration (default 100)
  --alpha ALPHA               coeefficient of data fidelity term
  --lambd LAMBD               coeefficient of regularization term
  --weights WEIGHTS           path to DnCNN weights
  --save_recon SAVE_RECON     file name for recoonstructed image
  --save_idx SAVE_IDX         file name for storing index
  --relax RELAX               relaxation for ADMM (from 0 to 1, default 0)
  --verbose
```

## Gaussian Deblurring
![Alt text](result/gaussian_deblur_1515.PNG?raw=true "Gaussian Deblurring")

```bash
python deblurr.py \
       --image imgs/tiger_224x224.png \
       --iter 10 \
       --sigma 1.5 \
       --alpha 1000 \
       --window 15 \
       --weights Denoisers/dnsr/DnCNN/weights/dncnn50_17.pth \
       --verbose
```

## Super-Resolution
![Alt text](result/depixelize_result.PNG?raw=true "Super Resolution")

```bash
python sr.py \
       --image imgs/tiger_224x224.png \
       --size 3 \
       --alpha 100 \
       --iter 100 \
       --prior dncnn \
       --weights Denoisers/dnsr/DnCNN/weights/dncnn50_17.pth \
       --verbose
```

# Installation

```bash
pip install -r requirements.txt
```

You can download addtional DnCNN weights here

[https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D](https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D)

## Installing pybm3d

To install pybm3d, first install Cython, then install manually by running setup.py
```bash
pip install Cython
git clone --recursive https://github.com/ericmjonas/pybm3d.git
cd pybm3d
python setup.py install
```

## Python Version
Python 3.9 is used for this code but older version most likely works, too.

# Usage

We use 2 objects to run Plug-and-Play ADMM.
- forward model
- denoiser

You can write your own forward model class and image prior class, but it must have the following methods.

```python
class Forward_Model:
  def __init__(self):
    # initialization if needed

  def prox(self, x):
    # return proximal operator at x

class Denoiser:
  def __init__(self):
    # initialization if needed

  def __call__(self, x):
    # denoise x for Gaussian noise
```
For consistency, images are normalized by dividing the pixels by 255 !!

## Example

```python
y = get_y() # your measurement
f = my_forward_model(y)
g = my_image_prior()
optimizer = PnP_ADMM(f, g, transform=t)
optimizer.init(np.random.rand(*y.shape))
x_hat, v_hat = optimizer.run(iter=100, return='both')
```

PnPADMM class takes the following as input argument
```python
class PnPADMM:
    def __init__(self, forward, denoiser):
      # initialization
      # forward: forward model object with prox() methods
      # denoiser: denoiser object with __call__ methods

    def init(self, v, u=None):
      # v: initial input of image_prior
      # u: initial value for variable u
      #    if not provided, use the same value as v
      #
      # for v, random initialization (range from 0 to 1)
      # is recommended.

    def run(self, iter=100, relax=0, return_value='both', verbose=False):
      # runs optimization
      # iter: iteration
      # relax: relaxation constant (range from 0 to 1, 0 if no relaxation)
      # return_value: if 'both', this will return both x and v,
      #               if 'x', this will only return x
      #               if else, this will only return v                
      # verbose: if true, prints iteration number
```
