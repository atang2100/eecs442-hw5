from utils import dehomogenize, homogenize, draw_epipolar, visualize_pcd
import numpy as np
import cv2
import pdb
import os

def normalize_points(pts):
    """
    Normalizes the points such that they have zero mean and unit standard deviation,
    and their average distance to the origin is sqrt(2)

    Inputs:
    - pts: Numpy array of shape (N,2) giving image coordinates

    Returns:
    - pts_norm: Numpy array of shape (N,2) giving the normalized image coordinates
    - T: Numpy array of shape (3,3) representing the normalization matrix
    """
    # Compute mean and standard deviation
    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0)

    # Scale points to have unit standard deviation and mean at the origin
    pts_centered = pts - mean
    pts_norm = pts_centered / std

    # Compute average distance to origin
    dist = np.sqrt(np.sum(pts_norm**2, axis=1))
    avg_dist = np.mean(dist)

    # Compute normalization matrix
    T = np.array([[1/avg_dist, 0, -mean[0]/avg_dist],
                  [0, 1/avg_dist, -mean[1]/avg_dist],
                  [0, 0, 1]])

    # Apply normalization matrix to points
    pts_norm_homogeneous = np.concatenate([pts_norm, np.ones((pts.shape[0], 1))], axis=1)
    pts_norm_homogeneous = np.dot(T, pts_norm_homogeneous.T).T[:, :2]

    return pts_norm_homogeneous, T

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
    print("correct answer is", FOpenCV)
    
    # Normalize the points
    # T1 = np.array([[2/shape[1], 0, -1],
    #                [0, 2/shape[0], -1],
    #                [0, 0, 1]])
    # T2 = np.array([[2/shape[1], 0, -1],
    #                [0, 2/shape[0], -1],
    #                [0, 0, 1]])
    # pts1_norm = np.dot(T1, np.vstack((pts1.T, np.ones(pts1.shape[0])))).T[:,:2]
    # pts2_norm = np.dot(T2, np.vstack((pts2.T, np.ones(pts2.shape[0])))).T[:,:2]

    # # Construct the A matrix for the linear system
    # A = np.zeros((pts1.shape[0], 9))
    # for i in range(pts1.shape[0]):
    #     x, y = pts1_norm[i,:]
    #     u, v = pts2_norm[i,:]
    #     A[i,:] = [x*u, y*u, u, x*v, y*v, v, x, y, 1]

    # # Solve the linear system to obtain F
    # U,S,V = np.linalg.svd(A)
    # F = V[-1,:].reshape(3,3)

    # # Enforce rank-2 constraint on F
    # U,S,V = np.linalg.svd(F)
    # S[-1] = 0
    # F = np.dot(U, np.dot(np.diag(S), V))

    # # Denormalize F
    # F = np.dot(np.dot(T2.T, F), T1)
    
    
    # Step 1: normalize the points
    pts1_norm, T1 = normalize_points(pts1, shape)
    pts2_norm, T2 = normalize_points(pts2, shape)

    # Step 2: compute the matrix A
    A = construct_A(pts1_norm, pts2_norm)

    # Step 3: solve for F_init using the eigenvector corresponding to the smallest eigenvalue of A^T A
    ATA = np.dot(A.T, A)
    w, V = np.linalg.eig(ATA)
    w_argsort = np.argsort(w)
    F_init = V[:, w_argsort[0]].reshape(3, 3)

    # Step 4: enforce rank-2 constraint using SVD
    U, s, Vt = np.linalg.svd(F_init)
    s[2] = 0
    F_rank2 = U @ np.diag(s) @ Vt

    # Step 5: denormalize the fundamental matrix
    F = T2.T @ F_rank2 @ T1

    F = F / F[-1, -1]




    # F = np.eye(3)
    print("final answer is", F)
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
        # e1, e2 = compute_epipoles(F)
        # print(e1, e2)
        # #to get the real coordinates, divide by the last entry
        # print(e1[:2]/e1[-1], e2[:2]/e2[-1])

        # outname = os.path.join(output, name + "_us.png")
        # # If filename isn't provided or is None, this plt.shows().
        # # If it's provided, it saves it
        # draw_epipolar(img1, img2, F, pts1[::10, :], pts2[::10, :],
        #               epi1=e1, epi2=e2, filename=outname)

        # if 0:
        #     #you can turn this on or off
        #     pcd = find_triangulation(K1, K2, F, pts1, pts2)
        #     visualize_pcd(pcd,
        #                   filename=os.path.join(output, name + "_rec.png"))


