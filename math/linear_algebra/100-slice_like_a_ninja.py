#!/usr/bin/env python3
""" function that slices a matrix along a specific axes
"""


def np_slice(matrix, axes={}):
    """ function : slices a matrix along a specific axes

    Arguments:
        matrix : matrix
        axes : dictionary where the key is an axis to slice along and
               the value is a tuple representing the slice to make along that
               axis

    Returns:
        new matrix
    """
    # Make a copy of the input matrix to avoid modifying the original
    result_matrix = matrix.copy()

    # Iterate over the axes specified in the 'axes' dictionary
    for axis, slice_tuple in axes.items():
        # Create a tuple for slicing using numpy.s_
        slices = [slice(None)] * len(result_matrix.shape)

        # Handle the case when only start is provided
        if len(slice_tuple) == 1:
            slices[axis] = slice(slice_tuple[0])
        # Handle the case when only start and stop are provided
        elif len(slice_tuple) == 2:
            slices[axis] = slice(slice_tuple[0], slice_tuple[1])
        # Handle the case when start, stop, and step are provided
        elif len(slice_tuple) == 3:
            slices[axis] = slice(slice_tuple[0], slice_tuple[1],
                                 slice_tuple[2])

        # Apply the slice to the corresponding axis
        result_matrix = result_matrix[tuple(slices)]

    return result_matrix
