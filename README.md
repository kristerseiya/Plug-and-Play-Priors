# What is Plug-and-Play Priors?
Plug-and-Play is a way of making the optimization of image restoration problem easier by decoupling the data fidelity term (forward model) and regularization term (image prior), thus allows you to plug in arbitrary gaussian denoisers for different image restoration tasks.

This is an attempted implementation of Plug-and-Play ADMM.

## Compressed Sensing
![Alt text](result/tiger_dncnn.png?raw=true "Compressed Sensing")

```bash
python compressed_sensing.py \
       --image imgs/tiger_224x224.png \
       --sample 0.2 \
       --prior dncnn \
       --alpha 2000 \
       --iter 100 \
       --weights DnCNN/dncnn50.pth \
       --verbose
```
```markdown
python compressed_sensing.py -h

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
       --weights DnCNN/dncnn50.pth \
       --verbose
```

## Depixelation
![Alt text](result/depixelize_result.PNG?raw=true "Depixelation")

```bash
python depixelate.py \
       --image imgs/tiger_224x224.png \
       --size 3 \
       --alpha 100 \
       --iter 100 \
       --prior dncnn \
       --weights DnCNN/dncnn50.pth \
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

We use 3 class objects to run Plug-and-Play ADMM.
- forward model
- image prior
- variable transformation [optional]

Variable transformation is a transformation between variables x and v where x is the input argument of forward model and v is the input argument of the prior model. The optimization will run the augmented Lagrangian of constrained optimization, transform(x) = v. If not explicitly given, it will assume x = v. (do not explicitly define variable transformation unless the transformation is invertible or one-to-one mapping!)

You can write your own forward model class and image prior class, but it must have the following attributes and methods
```python
class your_forward_or_prior_model:
  def __init__(self):
    # initialization if needed

  def prox(self, x):
    # this method computes the proximal operator at a given value x
    # for an image prior, output is simply a
    # Gaussian-denoised image
```
For consistency, images are normalized by dividing the pixels by 255 !!

## Examples
```python
class my_forward_model:
    def __init__(self, y):
      self.y = y # measurement
      # do other stuff

    def prox(self, x):
      # compute value of proximal operator at x
      return prox_output
```

```python
class my_image_prior:
    def __init__(self):
      # initialization if needed

    def prox(self, x):
      # compute value of proximal operator at x
      return prox_output
```

```python
class my_transform:
    def __init__(self):
      # initialization if needed

    def __call__(self, x):
      # map x to v
      return x_transformed

    def inverse(self, v):
      # map v to x
      return v_transformed
```

```python
y = get_y() # your measurement
f = my_forward_model(y)
g = my_image_prior()
t = my_transform()
optimizer = PnP_ADMM(f, g, transform=t)
optimizer.init(np.random.rand(*y.shape))
x_hat, v_hat = optimizer.run(iter=100, return='both')
```

PnP_ADMM class takes the following as input argument
```python
class PnP_ADMM:
    def __init__(self, forward, image_prior, transform=None):
      # initialization
      # forward: forward model object with set(), prox() methods
      # image_prior: image_prior object with set(), prox() methods
      # transform: transformation between x and v
      #            if not given, it will assume x = v
      #
      #  (do not explicitly define variable transformation unless
      #   the transformation is invertible or one-to-one mapping!)

    def init(self, v, u=None):
      # v: initial input of image_prior
      # u: initial value for variable u
      #    if not provided, use the same value as v
      #
      # using random values (range from 0 to 1) 
      # or constant array of 0.5 is recommended.

    def run(self, iter=100, relax=0, return_value='both', verbose=False):
      # runs optimization
      # iter: iteration
      # relax: relaxation constant (range from 0 to 1, 0 if no relaxation)
      # return_value: if 'both', this will return both x and v,
      #               if 'x', this will only return x
      #               if else, this will only return v                
      # verbose: if true, prints iteration number
```
