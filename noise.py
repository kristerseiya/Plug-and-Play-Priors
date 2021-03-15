
import numpy as np

def add_gauss(image, std=0.1):
    return image + np.random.normal(size=image.shape) * std

def add_poisson(image, lambd):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))

        # Generating noise for each unique value in image.
    return np.random.poisson(image * vals) / float(vals)

# from PIL import Image
# import numpy as np
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('-i', type=str, required=True)
# parser.add_argument('-o', type=str, required=True)
# parser.add_argument('--sigma', type=float, default=0.1)
# args = parser.parse_args()
#
# img = np.array(Image.open(args.i).convert('L'))
# img = img.astype(np.float64)
# img += np.random.normal(scale=args.sigma*255, size=img.shape)
# img[img > 255] = 255
# img[img < 0] = 0
# img = img.astype(np.uint8)
# Image.fromarray(img, 'L').save(args.o)
