# -------------------------------------------------------------------
# MORPHOLOGY
# -------------------------------------------------------------------

# Basic tool for finding indices in a NumPy array
def find(condition):
    res = np.nonzero(condition)
    return res
    """
    Find indices in an array satisfying a given condition.

    Parameters
    ----------
    condition : numpy.ndarray
        Boolean array indicating desired locations.

    Returns
    -------
    tuple
        Array indices where the condition is True.
    Example
    ------
    >>> idxs = find(img > 0.5)
    >>> print(type(idxs), len(idxs))
    <class 'tuple'> 2
    """
# helper function for finding pixel locations satisfy a condition. 

# Thresholding operation: pixels <= threshold become 0, pixels > threshold become 1
def biImg_by_threshold_leq(img, threshold):
    """
    Convert an image into a binary image using a lower threshold.

    Pixels with intensity values less than or equal to the threshold
    are assigned 0, while pixels above the threshold are assigned 1.

    Parameters
    ----------
    img : numpy.ndarray
        Input grayscale image.
    threshold : float
        Intensity threshold value.

    Returns
    -------
    numpy.ndarray
        Binary image.
    Example
    -------
    >>> binary_img = biImg_by_threshold_leq(img, threshold=0.5)
    >>> print(binary_img.min(), binary_img.max())
    0.0 1.0
    """
    output_img = np.copy(img)

    idxs_0 = find(img <= threshold)
    idxs_1 = find(img > threshold)

    output_img[idxs_0] = 0
    output_img[idxs_1] = 1

    return output_img


# Thresholding operation: pixels >= threshold become 0, pixels < threshold become 1
def biImg_by_threshold_geq(img, threshold):
    """
    Convert an image into a binary image using an upper threshold.

    Pixels greater than or equal to the threshold are assigned 0,
    while pixels below the threshold are assigned 1.

    Parameters
    ----------
    img : numpy.ndarray
        Input grayscale image.
    threshold : float
        Intensity threshold value.

    Returns
    -------
    numpy.ndarray
        Binary image.

    Example
    -------
    >>> binary_img = biImg_by_threshold_geq(img, threshold=0.5)
    >>> print(binary_img.min(), binary_img.max())
    0.0 1.0
    """
    output_img = np.copy(img)

    idxs_0 = find(img >= threshold)
    idxs_1 = find(img < threshold)

    output_img[idxs_0] = 0
    output_img[idxs_1] = 1

    return output_img

   
@nb.jit()
def erosion(input_np_array,
            input_list_of_points,
            minimal_pixel_value=0):
    """
    Perform morphological erosion on a binary image.

    Erosion removes boundary pixels and reduces foreground structures.
    The output value of each pixel is determined by the minimum value
    within the structuring element neighborhood.

    Parameters
    ----------
    input_np_array : numpy.ndarray
        Binary input image.
    input_list_of_points : list
        Structuring element coordinate offsets.
    minimal_pixel_value : int, optional
        Pixel value representing background. Default at 0. 

    Returns
    -------
    numpy.ndarray
        Eroded binary image.

    Example
    -------
    >>> eroded_img = erosion(binary_img, square_kernels[0])
    >>> print(type(eroded_img), eroded_img.shape)
    <class 'numpy.ndarray'> (3850, 4000)
    """
    input_np_array_shape = np.shape(input_np_array)
    output_np_array = np.zeros(input_np_array_shape)

    for i in range(input_np_array_shape[0]):
        for j in range(input_np_array_shape[1]):

            if input_np_array[i, j] == minimal_pixel_value:
                output_np_array[i, j] = minimal_pixel_value
                continue

            relevant_pixel_values = []

            for k in range(len(input_list_of_points)):

                m = i - input_list_of_points[k][1]
                n = j + input_list_of_points[k][0]

                if (m >= 0) and (m < input_np_array_shape[0]) and (n >= 0) and (n < input_np_array_shape[1]):
                    relevant_pixel_values.append(input_np_array[m, n])

            output_np_array[i, j] = min(relevant_pixel_values)

    return output_np_array


@nb.jit()
def dilation(input_np_array,
             input_list_of_points,
             maximal_pixel_value=1):
    """
    Perform morphological dilation on a binary image.

    Dilation expands foreground regions by assigning each pixel the
    maximum value found within the structuring element neighborhood.

    Parameters
    ----------
    input_np_array : numpy.ndarray
        Binary input image.
    input_list_of_points : list
        Structuring element coordinate offsets.
    maximal_pixel_value : int, optional
        Pixel value representing foreground. Default of 1. 

    Returns
    -------
    numpy.ndarray
        Dilated binary image.

    Example
    -------
    >>> dilated_img = dilation(binary_img, square_kernels[0])
    >>> print(type(dilated_img), dilated_img.shape)
    <class 'numpy.ndarray'> (3850, 4000)
    """
    input_np_array_shape = np.shape(input_np_array)
    output_np_array = np.zeros(input_np_array_shape)

    for i in range(input_np_array_shape[0]):
        for j in range(input_np_array_shape[1]):

            if input_np_array[i, j] == maximal_pixel_value:
                output_np_array[i, j] = maximal_pixel_value
                continue

            relevant_pixel_values = []

            for k in range(len(input_list_of_points)):

                m = i + input_list_of_points[k][1]
                n = j - input_list_of_points[k][0]

                if (m >= 0) and (m < input_np_array_shape[0]) and (n >= 0) and (n < input_np_array_shape[1]):
                    relevant_pixel_values.append(input_np_array[m, n])

            output_np_array[i, j] = max(relevant_pixel_values)

    return output_np_array


def opening(input_np_array,
            input_list_of_points):
    """
    Perform morphological opening.

    Opening consists of erosion followed by dilation. It removes small
    objects and noise while preserving larger structures.

    Parameters
    ----------
    input_np_array : numpy.ndarray
        Binary input image.
    input_list_of_points : list
        Structuring element coordinates.

    Returns
    -------
    numpy.ndarray
        Opened image.

   Example
   -------
   >>> opened_img = opening(binary_img, square_kernels[0])
    >>> print(type(opened_img), opened_img.shape)
    <class 'numpy.ndarray'> (3850, 4000)
    """
    return dilation(
        erosion(input_np_array, input_list_of_points),
        input_list_of_points
    )


def closing(input_np_array,
            input_list_of_points):
    """
    Perform morphological closing.

    Closing consists of dilation followed by erosion. It fills small
    holes and connects nearby foreground regions.

    Parameters
    ----------
    input_np_array : numpy.ndarray
        Binary input image.
    input_list_of_points : list
        Structuring element coordinates.

    Returns
    -------
    numpy.ndarray
        Closed image.

    Example
    -------
    >>> closed_img = closing(binary_img, square_kernels[0])
    >>> print(type(closed_img), closed_img.shape)
    <class 'numpy.ndarray'> (3850, 4000)
    """
    return erosion(
        dilation(input_np_array, input_list_of_points),
        input_list_of_points
    )

@nb.jit()
def get_rectangle_coordinates(input_np_array):
    """
    Generate coordinate offsets for a rectangular structuring element.

    This function creates a list of relative pixel coordinates centered
    around the origin of an input array. These coordinates define the
    neighborhood used by morphological operations such as erosion and
    dilation.

    Parameters
    ----------
    input_np_array : numpy.ndarray
        Array representing the desired size of the rectangular
        structuring element.

    Returns
    -------
    list
        List of numpy arrays containing coordinate offsets relative
        to the center pixel.

    Notes
    -----
    The center of the structuring element is treated as (0,0). For
    example, a 3x3 structuring element produces coordinates covering
    the entire neighborhood around the central pixel.

    Example
    -------
    A 3x3 structuring element generates offsets:

        [-1, -1]  [0, -1]  [1, -1]
        [-1,  0]  [0,  0]  [1,  0]
        [-1,  1]  [0,  1]  [1,  1]
    """
    input_np_array_shape = np.shape(input_np_array)
    output_list = []

    origin_i = int(input_np_array_shape[0] / 2)
    origin_j = int(input_np_array_shape[1] / 2)

    for i in range(input_np_array_shape[0]):
        for j in range(input_np_array_shape[1]):

            output_list.append(np.array([origin_j - j, origin_i - i]))

    return output_list


def get_horizontal_SE_list(maximal_SE_lengths):
    """
    Generate horizontal line structuring elements of increasing size.

    This function creates a collection of horizontal structuring elements
    with lengths ranging from 2 pixels up to the specified maximum length.

    Parameters
    ----------
    maximal_SE_lengths : int
        Maximum size of the horizontal structuring element.

    Returns
    -------
    list
        List of horizontal structuring elements. Each element contains
        coordinate offsets representing pixels along a horizontal line.

    Notes
    -----
    These structuring elements are used in morphological operations to
    analyze horizontally oriented structures at different spatial scales.

    Example
    -------
    >>> h_kernels = get_horizontal_SE_list(maximal_SE_lengths=5)
    >>> print(type(h_kernels), len(h_kernels))
    <class 'list'> 4
    """
    kernel_list = []

    for i in range(2, maximal_SE_lengths + 1):

        buffer_list = []
        buffer_list.append(np.array([0, 0]))

        counter = int(1)
        left_counter = int(-1)
        right_counter = int(1)

        while len(buffer_list) < i:

            if counter % 2 == 1:
                buffer_list.append(np.array([right_counter, 0]))
                counter = counter + 1
                right_counter = right_counter + 1

            else:
                buffer_list.append(np.array([left_counter, 0]))
                counter = counter + 1
                left_counter = left_counter - 1

        kernel_list.append(buffer_list)

    return kernel_list


def get_vertical_SE_list(maximal_SE_lengths):
    """
    Generate vertical line structuring elements of increasing size.

    Creates vertical structuring elements ranging from length 2 to the
    specified maximum length.

    Parameters
    ----------
    maximal_SE_lengths : int
        Maximum size of the vertical structuring element.

    Returns
    -------
    list
        List of vertical structuring elements represented as coordinate
        offsets.

    Notes
    -----
    These kernels allow morphological operations to examine vertically
    oriented image structures across multiple spatial scales.

    Example
    -------
    >>> v_kernels = get_vertical_SE_list(maximal_SE_lengths=5)
    >>> print(type(v_kernels), len(v_kernels))
    <class 'list'> 4
    """
    kernel_list = []

    for i in range(2, maximal_SE_lengths + 1):

        buffer_list = []
        buffer_list.append(np.array([0, 0]))

        counter = int(1)
        left_counter = int(-1)
        right_counter = int(1)

        while len(buffer_list) < i:

            if counter % 2 == 1:
                buffer_list.append(np.array([0, right_counter]))
                counter = counter + 1
                right_counter = right_counter + 1

            else:
                buffer_list.append(np.array([0, left_counter]))
                counter = counter + 1
                left_counter = left_counter - 1

        kernel_list.append(buffer_list)

    return kernel_list


def get_square_SE_list(maximal_SE_lengths):
    """
    Generate square structuring elements of increasing size.

    This function creates square neighborhoods used for morphological
    operations. Each square size produces a different spatial scale
    for filtration.

    Parameters
    ----------
    maximal_SE_lengths : int
        Maximum side length of the square structuring element.

    Returns
    -------
    list
        Collection of square structuring elements represented by
        coordinate offsets.

    Notes
    -----
    Increasing structuring element size allows morphological
    filtrations to capture features at progressively larger scales.

    Example
    -------
    >>> square_kernels = get_square_SE_list(maximal_SE_lengths=5)
    >>> print(type(square_kernels), len(square_kernels))
    <class 'list'> 4
    """
    kernel_list = []

    for i in range(2, maximal_SE_lengths + 1):
        kernel_list.append(
            get_rectangle_coordinates(input_np_array=np.zeros((i, i)))
        )

    return kernel_list

def img_to_1d_array(img):
    """
    Convert a two-dimensional image array into a one-dimensional array.

    Persistent homology libraries often require image data to be stored
    as a flattened vector. This function converts each row of the image
    into a single ordered array.

    Parameters
    ----------
    img : numpy.ndarray
        Two-dimensional image array.

    Returns
    -------
    numpy.ndarray
        Flattened one-dimensional representation of the image.

    Notes
    -----
    Pixel ordering is preserved by concatenating rows in reverse order
    relative to the input image.

    Example
    -------
    >>> flattened = flatten_img_reverse_rows(img)
    >>> print(flattened.ndim, flattened.shape[0] == img.size)
    1 True
    """
    result = []
    img_shape = np.shape(img)

    for i in range(img_shape[0]):
        result = list(img[i, :]) + result

    return np.array(result)


def persistence_of_img(img, maxdim=1):
    """
    Compute persistent homology of an image filtration.

    Calculates persistence diagrams describing the birth and death
    of topological features.

    Parameters
    ----------
    img : numpy.ndarray
        Input filtration image.
    maxdim : int, optional
        Maximum homology dimension calculated.

    Returns
    -------
    list
        Two persistence diagrams:
        - H0: connected components
        - H1: loops and holes

    Example
    -------
    >>> h0_h1_diagrams = compute_persistence(img, maxdim=1)
    >>> print(type(h0_h1_diagrams), len(h0_h1_diagrams))
    <class 'list'> 2
    """
    img = np.asarray(img, dtype=np.float64)

    if hasattr(cripser, "compute_ph"):
        ph = cripser.compute_ph(img, maxdim=maxdim)
    else:
        ph = cripser.computePH(img, maxdim=maxdim)

    persistence_0 = ph[ph[:, 0] == 0][:, 1:3]
    persistence_1 = ph[ph[:, 0] == 1][:, 1:3]

    return [persistence_0, persistence_1]

def persistence_of_morph_filtration(img,
                                    kernel_list,
                                    morph_type='closing'):
    """
    Compute persistent homology from a morphological filtration.

    Multiple morphological operations are applied using structuring
    elements of increasing size. The resulting filtration captures
    how image structures change across spatial scales.

    Parameters
    ----------
    img : numpy.ndarray
        Input binary image.
    kernel_list : list
        Collection of structuring elements.
    morph_type : str, optional
        Morphological operation:
        'opening', 'closing', 'erosion', or 'dilation'.

    Returns
    -------
    list
        Persistence diagrams for H0 and H1 features.

    Example
    -------
    >>> h0_h1_diagrams = compute_morphological_persistence(binary_img, square_kernels, morph_type='opening')
    >>> print(type(h0_h1_diagrams), len(h0_h1_diagrams))
    <class 'list'> 2
    """
    img_shape = np.shape(img)
    img_buff = np.zeros(img_shape)
    img_buff = img_buff + img

    for the_kernel in kernel_list:

        if morph_type == 'opening':
            morphed_img = opening(
                input_np_array=img,
                input_list_of_points=the_kernel
            )

        elif morph_type == 'closing':
            morphed_img = closing(
                input_np_array=img,
                input_list_of_points=the_kernel
            )

        elif morph_type == 'erosion':
            morphed_img = erosion(
                input_np_array=img,
                input_list_of_points=the_kernel
            )

        elif morph_type == 'dilation':
            morphed_img = dilation(
                input_np_array=img,
                input_list_of_points=the_kernel
            )

        else:
            raise ValueError(
                "morph_type must be 'opening', 'closing', "
                "'erosion', or 'dilation'"
    )

        img_buff = img_buff + morphed_img


    PDs = persistence_of_img(img_buff)

    return PDs

