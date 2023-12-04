#!/usr/bin/env python3
""" function to concatenate two arrays
"""


def cat_arrays(arr1, arr2):
    """ function : concatenate two array

    Arguments:
        arr1 : first array
        arr2 : second array

    Returns:
        new array
    """
    len1 = len(arr1)
    len2 = len(arr2)

    new_array = []

    for i in range(len1):
        new_array.append(arr1[i])
    for j in range(len2):
        new_array.append(arr2[j])

    return new_array
