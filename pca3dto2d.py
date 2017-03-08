# Group Members: 1) Reshma Bhatia (8959806814)   2) Shiv Prathik Velagala (2654019972)

import numpy as np

#np.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold= None)
np.set_printoptions(precision=15)
np.set_printoptions(suppress=True)

class PCA(object):

    def transform(self, X):

        N, D = X.shape
        self.N = N
        self.D = D


        cov_mat = np.cov([X[:, 0], X[:, 1], X[:, 2]])
        cov_mat = cov_mat.T
        #print('Cov Mat:', cov_mat)


        matrix_w = self.calEigen(cov_mat)

        transformed = matrix_w.T.dot(X.T)

        return transformed.T


    def calEigen(self, cov_mat):

        eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

        for ev in eig_vec_cov:
            np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))


        eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))]

        eig_pairs.sort(key=lambda x: x[0], reverse=True)


        #print("Sorted Eigen Values:")
        #for i in eig_pairs:
        #    print(i[0])

        print("Directions:")
        print eig_pairs[0][1], eig_pairs[1][1]


        mat_w = np.hstack((eig_pairs[0][1].reshape(self.D, 1), eig_pairs[1][1].reshape(self.D, 1)))
       #print('Mat W:', mat_w)
        return mat_w


if __name__ == "__main__":

    X = np.genfromtxt('pca-data.txt')

    N, D = X.shape

    #print ("Original: \n", X)

    pca3dto2d = PCA()

    transformed = pca3dto2d.transform(X)

    print ("Data points transformed to 2D:", transformed)

