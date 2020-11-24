""" Make an homography between two images.

Make an homography between two images. In this script we use automatic methods
to select and identify keypoints. The available method so far is ORB.

"""
import cv2
from matplotlib import pyplot as plt
import numpy as np
from visio_per_computador.common import descriptors


def run():
    """ Read two images of a document and tries to warpped them.
    """
    method_kp_desc = "fast_brief"

    method_match = "BF_H_k"
    k_match = 2#if 0 it will not use knn, and filtering will be done via min_distance

    method_filter_matches = "KNN"
    min_d = 40
    proportion = 0.65

    # We read the images
    img_L = cv2.imread("../../in/descriptors/canon_L.jpg")
    img_L_g = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)

    img_R = cv2.imread("../../in/descriptors/canon_R.jpg")
    img_R_g = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)


    kp1, desc1 = descriptors.get_kp_desc(method=method_kp_desc, img=img_L_g)
    kp2, desc2 = descriptors.get_kp_desc(method=method_kp_desc, img=img_R_g)



    #img_L_g = cv2.drawKeypoints(img_L_g, kp1, img_L_g, color=(255, 0, 0))
    #plt.imshow(img_L_g)
    #plt.show()

    if k_match == 0:
        matches = descriptors.match_descriptors(method=method_match, desc1=desc1, desc2=desc2,)
    else:
        matches = descriptors.match_descriptors(method=method_match, desc1=desc1, desc2=desc2, k=k_match)

    #Draw the matches without filtering
    #if k_match==0:
    #    res = cv2.drawMatches(img_L_g, kp1, img_R_g, kp2,
     #                         matches, None,
      #                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #else:
    #    res = cv2.drawMatches(img_L_g, kp1, img_R_g, kp2,
    #                         [m[0] for m in matches], None,
    #                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #plt.imshow(res)
    #plt.show()

    matches = descriptors.filter_matches(method=method_filter_matches, matches=matches, min_distance=min_d, proportion=proportion)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        if k_match == 0:
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt
        else:
            points1[i, :] = kp1[match[0].queryIdx].pt
            points2[i, :] = kp2[match[0].trainIdx].pt

    #Draw the matches after filtering

    #if k_match==0:
    #    res = cv2.drawMatches(img_L_g, kp1, img_R_g, kp2,
    #                          matches, None,
    #                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #else:
    #    res = cv2.drawMatches(img_L_g, kp1, img_R_g, kp2,
    #                          [m[0] for m in matches], None,
    #                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #plt.imshow(res)
    #plt.show()

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography
    height, width = img_R_g.shape
    im1Reg = cv2.warpPerspective(img_L_g, h, (width, height))

    plt.subplot(1, 3, 1)
    plt.title("Warped")
    plt.imshow(im1Reg, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.imshow(img_L_g, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.imshow(img_R_g, cmap="gray")

    plt.show()

    cv2.imwrite("../out/document_fixed.png", im1Reg)

    dst = cv2.warpPerspective(img_R,h,(img_L.shape[1] + img_R.shape[1], img_L.shape[0]))
    plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
    plt.show()
    plt.figure()
    dst[0:img_L.shape[0], 0:img_L.shape[1]] = img_L
    cv2.imwrite('../out/output.jpg',dst)
    plt.imshow(dst)
    plt.show()

run()
