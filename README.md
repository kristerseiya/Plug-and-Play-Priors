## Description
This is an attempt of implementation of Plug-and-Play ADMM.

## Installation

```bash
pip install -r requirements.txt
```

You can download DnCNN weights here

[https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D](https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D)

# Usage

You need 3 objects to run Plug-and-Play ADMM.
- forward model
- image prior
- variable transformation

Variable transformation is a transformation between variables x and v where x is the input argument of forward model and v is the input argument of the prior model. The optimization will run the augmented Lagrangian of constrained optimization, transform(x) = v.

You can write your own forward model class and image prior class, but it must have the following attributes and methods
```python
class your_forward_or_prior_model:
  def __init__(self):
    self.input_shape = # input shape
    # this stores the shape of accepted variable.
    # this is used to initialize some variables inside the optimization algorithm.

  def set(self, alpha):
    # this takes the Lagrangian multiplier.
    # this is called at the initial stage of optimization.

  def prox(self, x):
    # this computes the proximal operator at a given value x
```
For consistency, images are normalized by dividing the pixels by 255 !!

## Examples
```python
class my_forward_model:
    def __init__(self, y, input_shape):
      self.y = y # measurement
      self.input_shape = input_shape
      # do other stuff

    def set(self, alpha):
      self.alpha = alpha
      # do other stuff

    def prox(self, x):
      # compute value of proximal operator at x
      return prox_output
```

```python
class my_image_prior:
    def __init__(self, input_shape):
      self.input_shape = input_shape
      # initialization if needed

    def set(self, alpha):
      self.alpha = alpha
      # do other stuff

    def prox(self, x):
      # compute value of proximal operator at x
      return prox_output
```

```python
class my_transform:
    def __init__(self, x_shape, v_shape):
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
f = my_forward_model()
g = my_image_prior()
t = my_transform()
optimizer = PnP_ADMM(f, g, transform=t)
x_hat, v_hat = optimizer.run(alpha=1e-2., iter=100)
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

    def run(self, alpha=1e-2, iter=100, verbose=True, return_value='both'):
      # runs optimization
      # alpha: Lagrangian multiplier, theoretically this should not affect the performance of convex optimization
      # iter: iteration
      # verbose: if true, prints iteration number
      # return_value: if 'both', this will return both x and v,
      #               if 'x', this will only return x
      #               if else, this will only return v                
```
