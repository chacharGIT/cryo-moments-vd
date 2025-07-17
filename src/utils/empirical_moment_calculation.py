import numpy as np
from aspire.image import Image

def first_empirical_moment(images: Image):
    return np.mean(images._data, axis=0)


def second_empirical_moment(images: Image):
    data = images._data 
    num_images, height, width = data.shape

    # Compute the outer product for each image: (h, w) ⊗ (h, w) → (h, w, h, w)
    second_moment = np.einsum("nij,nkl -> ijkl", data, data) / num_images

    return second_moment
