## Description
This is an attempt of implementation of Plug-and-Play ADMM.

## Installation

```bash
pip install -r requirements
```

# Usage

You need 3 objects to run Plug-and-Play ADMM.
- forward model
- image prior
- variable transformation

Variable transformation is a transformation between variables x and v where x is the input argument of forward model and v is the input argument of the prior model. The optimization will run the augmented Lagrangian of constrained optimization such that transform(x) = v.

You can write your own forward model class and image prior class, but it must have the following methods
- self.set(alpha) that takes the Lagrangian multiplier. This is called at the initial stage of optimization.
- self.prox(x) that computes the proximal operator at a given value x


## Examples
```python
class my_forward_model:
    def __init__(self, y):
      self.y = y # measurement
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
    def __init__(self):
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
f = my_forward_model()
g = my_image_prior()
t = my_transform()
optimizer = PnP_ADMM(f, g, variable_shape=y.shape, transform=t)
x_hat, v_hat = optimizer.run(alpha=100., iter=100)
```

PnP_ADMM class takes the following as input argument
```python
class PnP_ADMM:
    def __init__(self, forward, image_prior, variable_shape, transform=None):
      # initialization
      # forward: forward model object with set(), prox() methods
      # image_prior: image_prior object with set(), prox() methods
      # variable_shape: input shape of image_prior.prox() method
      # transform: transformation between x and v

    def run(self, alpha=100, iter=100, verbose=True, return_value='both'):
      # runs optimization
      # alpha: Lagrangian multiplier, theoretically this should not affect the performance of convex optimization
      # iter: iteration
      # verbose: if true, prints iteration number
      # return_value: if 'both', this will return both x and v,
      #               if 'x', this will only return x
      #               if else, this will only return v                
```
