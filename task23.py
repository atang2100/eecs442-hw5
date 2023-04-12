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
    print("correct answer is\n", FOpenCV)
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
    
    print("my answer is\n", F)
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
    pcd = None
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    # Compute essential matrix E from F and K1, K2
    E = np.dot(K2.T, np.dot(F, K1))
    # Decompose E into four possible camera matrices for M2
    R1, R2, t = cv2.decomposeEssentialMat(E)
    K0 = np.eye(3)
    M1 = np.hstack([K1, np.zeros((3,1))])
    pcd = np.zeros((pts1.shape[0], 4))
    max_inliers = 0
    for R, T_sign in [(R1, 1), (R1, -1), (R2, 1), (R2, -1)]:
        M2 = np.hstack([np.dot(K2, R), T_sign*t])
        # Triangulate points using M1 and M2
        homog_pts_3d = cv2.triangulatePoints(M1, M2, pts1.T, pts2.T)
        # Convert to inhomogeneous coordinates
        inhomog_pts_3d = homog_pts_3d / homog_pts_3d[3,:]
        # Count number of inlier points (positive z)
        num_inliers = np.sum(inhomog_pts_3d[2,:] > 0)
        # Update max inliers and pcd if necessary
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            pcd = inhomog_pts_3d.T
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
        print(e1, e2)
        #to get the real coordinates, divide by the last entry
        print(e1[:2]/e1[-1], e2[:2]/e2[-1])

        outname = os.path.join(output, name + "_us.png")
        # If filename isn't provided or is None, this plt.shows().
        # If it's provided, it saves it
        draw_epipolar(img1, img2, F, pts1[::10, :], pts2[::10, :],
                      epi1=e1, epi2=e2, filename=outname)

        if 0:
            #you can turn this on or off
            pcd = find_triangulation(K1, K2, F, pts1, pts2)
            visualize_pcd(pcd,
                          filename=os.path.join(output, name + "_rec.png"))


