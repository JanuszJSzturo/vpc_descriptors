# -*- coding: utf-8 -*-
""" Module containing a set of function to work with descriptors

This module contains three functions that allow, respectively: get the
descriptors and the keypoints, match descriptors and filter the matches.

Multiple methods are available in each function.

"""
import cv2
import numpy as np


def get_kp_desc(method: str, img: np.ndarray, **kwargs):
    """ Gets the keypoints and the descriptors from an image.

    Calculates the keypoints of an image and its descriptors. To do so we use
    different well-known algorithms. The method used is passed as parameter.
    The results are always a tuple with two elements, the keypoints and its
    descriptors


    Args:
        method (str): See below for more information
        img (np.ndarray): Image to extract the descriptors
        **kwargs: Extra arguments for the methods
    Methods:
        "O" (ORB): Oriented FAST and rotated BRIEF descriptors
        "sift": Scale-Invariant Feature Transform descriptors
        "surf": Speeded Up Robust Features
        "fast_brief": fast (Features from Accelerated Segment Test) keypoints
                      brief( Binary Robust Independent Elementary Features) descriptors
    Returns:
        Tuple with the descriptors and the descriptions
    """
    method_call = None

    if method == "O":
        orb = cv2.ORB_create(**kwargs)
        method_call = orb.detectAndCompute
    elif method == "sift":
        sift = cv2.xfeatures2d.SIFT_create(**kwargs)
        method_call = sift.detectAndCompute
    elif method == "surf":
        surf = cv2.xfeatures2d.SURF_create(**kwargs)
        method_call = surf.detectAndCompute

    if method == "fast_brief":
        fast = cv2.FastFeatureDetector_create(**kwargs)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(**kwargs)
        kp = fast.detect(img,None)
        kp, descs = brief.compute(img, kp)
    else:
        kp, descs = method_call(img, mask=None, **kwargs)

    return kp, descs


def match_descriptors(method: str, desc1, desc2, **kwargs):
    """ Search matches between two sets of descriptors

    Search matches between two set of descriptors. Multiple methods are
    available. The usefulness of the methods depend on the format of the
    descriptors.

    Args:
        method (str): See options below
        desc1: List of descriptors
        desc2: List of descriptors
    Methods:
        "D": Bruteforce matcher with Hamming.
        "F": Bruteforce without Hamming
    Returns:
        List of matches
    """
    matches = None
    if method == "D":
        matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(desc1, desc2, **kwargs)
    elif method == "F":
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(desc1, desc2, **kwargs)
    elif method == "TEST":
        matcher = cv2.BFMatcher()
        matches = matcher.match(desc1, desc2)
    elif method == "D_KNN":
        matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.knnMatch(desc1, desc2, **kwargs)

    return matches


def filter_matches(method: str, matches, min_distance: int = None,
                   proportion: float = None):
    """ Filter matches by multiple conditions.

    Args:
        method (str): See below list of methods available
        matches: List of matches
        min_distance (int): Minimum distance to accept a match (DIST)
        proportion (float): Proportion between the second and first match
                            distances (KNN)

    Methods:
        "DIST": Select the matches with a distance higher than a minimum
        "KNN": Select the matches with a small distance on the first match and
                with a high value on the second. Defined by proportion

    Returns:
        List of filtered matches
    """
    if method == "DIST":
        matches = list(filter(lambda m: m.distance < min_distance, matches))
    elif method == "KNN":
        matches = list(filter(lambda m: m[0].distance < m[1].distance * proportion, matches))

    return matches
