# Overview 

This repository is a set of utilities and examples for the *Visió per Computador* subject of the *Master of Intelligent Systems* of the [UIB](https://www.uib.cat). 

## Install

We can install this repository with or without conda.

The first step is to download the repository
```
     git clone https://github.com/miquelmn/visio_per_computador.git
```
**Without conda**
```
     pip install -r requirements.txt
```
**With conda**
```
    conda env create --file eviroment.yml
```

## Content


### Homographies

A simple example of homographies. We select four points of two images to make an homography and warp one image to the plane of the other. The points are selected with a GUI based on ``matplotlib``.

### Descriptors
An example for the use of descriptors: scanning a document. To do so we extract the descriptors from two images and we make an homography.

We develop a small module (``common\descriptors.py``)with a set of functions to get the keypoints and descriptors, match two sets of descriptors and filter the matches
