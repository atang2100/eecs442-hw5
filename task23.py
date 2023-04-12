from utils import dehomogenize, homogenize, draw_epipolar, visualize_pcd
import numpy as np
import cv2
import pdb
import os
import math

def find_fundamental_matrix(shape, pts1, pts2):
    """
    Computes Fundamental Matrix F that relates points in two images by the:

        [u' v' 1] F [u v 1]^T = 0
        or
        l = F [u v 1]^T  -- the epipolar line for point [u v] in image 2
        [u' v' 1] F = l'   -- the epipolar line for point [u' v'] in image 1

    Where (u,v) and (u',v') are the 2D image coordinates of the left and
    the right images respectively.

    Inputs:
    - shape: Tuple containing shape of img1
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    """

    #This will give you an answer you can compare with
    #Your answer should match closely once you've divided by the last entry
    FOpenCV, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    # print("correct answer is\n", FOpenCV)
    ################################################################################################################################
    # make T
    s = np.maximum(pts1.max(), pts2.max())
    T = np.array([[1/s, 0, -1/2], [0, 1/s, -1/2], [0, 0, 1]])

    # compute scaled points
    pts1_homogeneous = homogenize(pts1)
    pts2_homogeneous = homogenize(pts2)

    p1 = (T @ pts1_homogeneous.T).T
    p2 = (T @ pts2_homogeneous.T).T

    # solve for U
    U = np.vstack([np.kron(element, p2[index]) for index, element in enumerate(p1)])

    # use eigenvectors/values to get F_init
    eigenvalues, eigenvectors = np.linalg.eig(U.T @ U)
    F_init = eigenvectors[:, np.argmin(eigenvalues)].reshape(3, 3)
    
    # rank reduce Finit using SVD, set last entry to 0
    U, S, V = np.linalg.svd(F_init)
    S[-1] = 0

    # compute F rank2
    F_rank2 = U @ np.diag(S) @ V

    # return T transpose @ Frank2 @ T, normalize with last entry
    F = T.T @ F_rank2 @ T
    F = F / F[2, 2]
    F = F.T
    
    # print("my answer is\n", F)
    return F


def compute_epipoles(F):
    """
    Given a Fundamental Matrix F, return the epipoles represented in
    homogeneous coordinates.

    Check: e2@F and F@e1 should be close to [0,0,0]

    Inputs:
    - F: the fundamental matrix

    Return:
    - e1: the epipole for image 1 in homogeneous coordinates
    - e2: the epipole for image 2 in homogeneous coordinates
    """
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    # Compute the left nullspace of F
    F = F.astype(np.float64)
    _, _, v = np.linalg.svd(F)
    e1 = v[-1] / v[-1][-1]  # Normalize to homogeneous coordinates

    # Compute the right nullspace of F
    _, _, u = np.linalg.svd(F.T)
    e2 = u[-1] / u[-1][-1]  # Normalize to homogeneous coordinates
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return e1, e2


def find_triangulation(K1, K2, F, pts1, pts2):
    """
    Extracts 3D points from 2D points and camera matrices. Let X be a
    point in 3D in homogeneous coordinates. For two cameras, we have

        p1 === M1 X
        p2 === M2 X

    Triangulation is to solve for X given p1, p2, M1, M2.

    Inputs:
    - K1: Numpy array of shape (3,3) giving camera instrinsic matrix for img1
    - K2: Numpy array of shape (3,3) giving camera instrinsic matrix for img2
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - pcd: Numpy array of shape (N,4) giving the homogeneous 3D point cloud
      data
    """
    max_num_positive_points = 0
    # cast everything to floats
    K1 = K1.astype(np.float64)
    K2 = K2.astype(np.float64)
    F = np.float64(F)
    pts1 = np.float64(pts1)
    pts2 = np.float64(pts2)
    
    # make E
    E = K2.T @ F @ K1
    R1, R2, t = cv2.decomposeEssentialMat(E)

    # M1 is 3 x 4, create all possible M2s
    M1 = np.float64(np.hstack((K1, np.zeros((3, 1)))))
    M2_1 = np.float64(np.hstack((np.dot(K2, R1), np.dot(K2, t))))
    M2_2 = np.float64(np.hstack((np.dot(K2, R1), np.dot(K2, -t))))
    M2_3 = np.float64(np.hstack((np.dot(K2, R2), np.dot(K2, t))))
    M2_4 = np.float64(np.hstack((np.dot(K2, R2), np.dot(K2, -t))))
    M2_options = [M2_1, M2_2, M2_3, M2_4]
    
    for M2 in M2_options:
        # 4d points
        homog_points = np.float64(cv2.triangulatePoints(M1, M2, pts1.T, pts2.T))
        # check number of positives in last row
        num_positive_points = np.sum(homog_points[3] > 0)
        if num_positive_points > max_num_positive_points:
            max_num_positive_points = num_positive_points
            best_X = homog_points
    # divide by last row
    best_X /= best_X[3]
    pcd = best_X
    return pcd


if __name__ == '__main__':

    # You can run it on one or all the examples
    names = os.listdir("task23")
    output = "results/"

    if not os.path.exists(output):
        os.mkdir(output)

    for name in names:
        print(name)

        # load the information
        img1 = cv2.imread(os.path.join("task23", name, "im1.png"))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(os.path.join("task23", name, "im2.png"))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        data = np.load(os.path.join("task23", name, "data.npz"))
        pts1 = data['pts1'].astype(float)
        pts2 = data['pts2'].astype(float)
        K1 = data['K1']
        K2 = data['K2']
        shape = img1.shape

        # compute F
        F = find_fundamental_matrix(shape, pts1, pts2)
        # compute the epipoles
        e1, e2 = compute_epipoles(F)
        # print(e1, e2)
        #to get the real coordinates, divide by the last entry
        # print(e1[:2]/e1[-1], e2[:2]/e2[-1])

        outname = os.path.join(output, name + "_us.png")
        # If filename isn't provided or is None, this plt.shows().
        # If it's provided, it saves it
        draw_epipolar(img1, img2, F, pts1[::10, :], pts2[::10, :],
                      epi1=e1, epi2=e2, filename=outname)

        if 1:
            #you can turn this on or off
            if name == "reallyInwards":
                E = K2.T @ F @ K1
                print("E is\n", E)
            pcd = find_triangulation(K1, K2, F, pts1, pts2)
            visualize_pcd(pcd,
                          filename=os.path.join(output, name + "_rec.png"))


