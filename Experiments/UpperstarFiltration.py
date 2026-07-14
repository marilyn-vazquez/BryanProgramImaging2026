# -*- coding: utf-8 -*-
import cripser

def compute_upper_star(preprocessed_img):
    """
    Compute an upper-star persistence diagram from a
    preprocessed grayscale image.
    """

    inverted_img = (
        preprocessed_img.max()
        - preprocessed_img
    )

    ph_upper = cripser.compute_ph(
        inverted_img.astype(float),
        maxdim=1
    )

    return ph_upper, inverted_img

