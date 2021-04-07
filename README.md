# What is Plug-and-Play Priors?
Plug-and-Play is a way of making the optimization of image restoration problem easier by decoupling the data fidelity term (forward model) and regularization term (image prior), thus allows you to plug in arbitrary gaussian denoisers for different image restoration tasks.

This is an attempted implementation of Plug-and-Play ADMM.

## Compressed Sensing
![Alt text](result/tiger_dncnn.png?raw=true "Compressed Sensing")

## Gaussian Deblurring
![Alt text](result/gaussian_deblur_1515.PNG?raw=true "Gaussian Deblurring")

## Depixelation
![Alt text](result/depixelize_result.PNG?raw=true "Depixelation")

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
  def __init__(self, input_shape):
    self.input_shape = input_shape
    # this variable stores the shape of accepted variable.
    # this is used to initialize some variables inside the optimization algorithm.

  def __call__(self, x):
    # this method computes the proximal operator at a given value x
    # for an image prior, output is the estimate of Gaussian-denoised image
```
For consistency, images are normalized by dividing the pixels by 255 !!

## Examples
```python
class my_forward_model:
    def __init__(self, y, input_shape):
      self.y = y # measurement
      self.input_shape = input_shape
      # do other stuff

    def __call__(self, x):
      # compute value of proximal operator at x
      return prox_output
```

```python
class my_image_prior:
    def __init__(self, input_shape):
      self.input_shape = input_shape
      # do other stuff

    def __call__(self, x):
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
f = my_forward_model(y, x_shape)
g = my_image_prior(v_shape)
t = my_transform()
optimizer = PnP_ADMM(f, g, transform=t)
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
      #  (do not explicitly define variable transformation unless the transformation is invertible or one-to-one mapping!)

    def run(self, iter=100, verbose=True, return_value='both'):
      # runs optimization
      # iter: iteration
      # verbose: if true, prints iteration number
      # return_value: if 'both', this will return both x and v,
      #               if 'x', this will only return x
      #               if else, this will only return v                
```
