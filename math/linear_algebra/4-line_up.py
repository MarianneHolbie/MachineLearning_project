#!/usr/bin/env python3
""" function to add two arrays"""


def add_arrays(arr1, arr2):
    """ function : add two array

    Arguments:
        arr1 : first array
        arr2 : second array

    Returns:
        new list
    """
    if len(arr1) != len(arr2):
        return None

    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result
