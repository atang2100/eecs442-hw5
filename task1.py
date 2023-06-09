import numpy as np
import utils


def find_projection(pts2d, pts3d):
    """
    Computes camera projection matrix M that goes from world 3D coordinates
    to 2D image coordinates.

    [u v 1]^T === M [x y z 1]^T

    Where (u,v) are the 2D image coordinates and (x,y,z) are the world 3D
    coordinates

    Inputs:
    - pts2d: Numpy array of shape (N,2) giving 2D image coordinates
    - pts3d: Numpy array of shape (N,3) giving 3D world coordinates

    Returns:
    - M: Numpy array of shape (3,4)

    """

    # M = None
    # ###########################################################################
    # # TODO: Your code here                                                    #
    # ###########################################################################

    # make matrix A
    N = pts2d.shape[0]
    A = np.zeros((2*N, 12))
    for i in range(N):
        X, Y, Z = pts3d[i]
        u, v = pts2d[i]
        A[2*i] = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
        A[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]

    # use SVD for projection matrix M
    _, _, VT = np.linalg.svd(A)
    M = VT[-1].reshape(3, 4)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return M

def compute_distance(pts2d, pts3d):
    """
    use find_projection to find matrix M, then use M to compute the average 
    distance in the image plane (i.e., pixel locations) 
    between the homogeneous points M X_i and 2D image coordinates p_i

    Inputs:
    - pts2d: Numpy array of shape (N,2) giving 2D image coordinates
    - pts3d: Numpy array of shape (N,3) giving 3D world coordinates

    Returns:
    - float: a average distance you calculated (threshold is 0.01)

    """
    # distance = None
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    # get projection matrix M from find_projection
    M = find_projection(pts2d, pts3d)
    num_pts = pts2d.shape[0]
    total_distance = 0

    for i in range(num_pts):
        # Convert 3D point to homogeneous coords
        X_i = np.hstack((pts3d[i], 1))

        # project 3D point to 2D
        P_i = np.dot(M, X_i)
        P_i /= P_i[2]

        # get distance between projected point and actual 2D point using 2norm
        distance = np.linalg.norm(P_i[:2] - pts2d[i])
        total_distance += distance

    # divide by the number of points
    distance = total_distance / num_pts
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distance

if __name__ == '__main__':
    pts2d = np.loadtxt("task1/pts2d.txt")
    pts3d = np.loadtxt("task1/pts3d.txt")

    M = find_projection(pts2d, pts3d)
    print("M is\n", M)


    # Alternately, for some of the data, we provide pts1/pts1_3D, which you
    # can check your system on via
    """
    data = np.load("task23/ztrans/data.npz")
    pts2d = data['pts1']
    pts3d = data['pts1_3D']
    """
   
    foundDistance = compute_distance(pts2d, pts3d)
    print("Distance: %f" % foundDistance)
