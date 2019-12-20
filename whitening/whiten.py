import os
import numpy as np

def whitenapply(X, m, P, dimensions=None):
    
    if not dimensions:
        dimensions = P.shape[0]

    X = np.dot(P[:dimensions, :], X-m)
    X = X / (np.linalg.norm(X, ord=2, axis=0, keepdims=True) + 1e-6)

    return X



def pcawhitenlearn(X):

    N = X.shape[1]

    # Learning PCA w/o annotations
    m = X.mean(axis=1, keepdims=True)
    Xc = X - m
    Xcov = np.dot(Xc, Xc.T)
    Xcov = (Xcov + Xcov.T) / (2*N)
    eigval, eigvec = np.linalg.eig(Xcov)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P = np.dot(np.linalg.inv(np.sqrt(np.diag(eigval))), eigvec.T)
    
    return m, P

def whitenlearn(X, qidxs, pidxs):

    # Learning Lw w annotations
    m = X[:, qidxs].mean(axis=1, keepdims=True)
    df = X[:, qidxs] - X[:, pidxs]
    S = np.dot(df, df.T) / df.shape[1]
    P = np.linalg.inv(cholesky(S))
    df = np.dot(P, X-m)
    D = np.dot(df, df.T)
    eigval, eigvec = np.linalg.eig(D)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P = np.dot(eigvec.T, P)

    return m, P





def DynP_Cs_whitenlearn(X, dyn_p_ms, qidxs, pidxs):

    dyn_p = np.mean(dyn_p_ms,axis=0)

    # Learning Lw w annotations
    m = X[:, qidxs].mean(axis=1, keepdims=True)
    df = X[:, qidxs] - X[:, pidxs]

    # Sort the sample pair based on dynamic p.
    # Smaller p is better.
    df_dyn_p = dyn_p[qidxs] + dyn_p[pidxs]
    id_sort = np.argsort(df_dyn_p)
    df_sort = df[:,id_sort]

    S = np.dot(df, df.T) / df.shape[1]
    P0 = np.linalg.inv(cholesky(S))

    df = np.dot(P0, X-m)
    D = np.dot(df, df.T)
    eigval, eigvec = np.linalg.eig(D)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P = np.dot(eigvec.T, P0)

    # ----------------------------
    # X:90% (Only use the top 90% of the positive sample pairs for the projection)
    df_90 = df_sort[:,0:int(0.9*np.shape(df)[1])]
    S = np.dot(df_90, df_90.T) / df_90.shape[1]
    P0 = np.linalg.inv(cholesky(S))

    df = np.dot(P0, X-m)
    D = np.dot(df, df.T)
    eigval, eigvec = np.linalg.eig(D)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P_90 = np.dot(eigvec.T, P0)

    # ----------------------------
    # X:80% (Only use the top 80% of the positive sample pairs for the projection)
    df_80 = df_sort[:,0:int(0.8*np.shape(df)[1])]
    S = np.dot(df_80, df_80.T) / df_80.shape[1]
    P0 = np.linalg.inv(cholesky(S))

    df = np.dot(P0, X-m)
    D = np.dot(df, df.T)
    eigval, eigvec = np.linalg.eig(D)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P_80 = np.dot(eigvec.T, P0)

    # ----------------------------
    # X:50% (Only use the top 50% of the positive sample pairs for the projection)
    df_50 = df_sort[:,0:int(0.5*np.shape(df)[1])]
    S = np.dot(df_50, df_50.T) / df_50.shape[1]
    P0 = np.linalg.inv(cholesky(S))

    df = np.dot(P0, X-m)
    D = np.dot(df, df.T)
    eigval, eigvec = np.linalg.eig(D)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P_50 = np.dot(eigvec.T, P0)

    # ----------------------------
    # X:30% (Only use the top 30% of the positive sample pairs for the projection)
    df_30 = df_sort[:,0:int(0.3*np.shape(df)[1])]
    S = np.dot(df_30, df_30.T) / df_30.shape[1]
    P0 = np.linalg.inv(cholesky(S))

    df = np.dot(P0, X-m)
    D = np.dot(df, df.T)
    eigval, eigvec = np.linalg.eig(D)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P_30 = np.dot(eigvec.T, P0)

    return m, P, P_90, P_80, P_50, P_30








def cholesky(S):
    # Cholesky decomposition
    # with adding a small value on the diagonal
    # until matrix is positive definite
    alpha = 0
    while 1:
        try:
            L = np.linalg.cholesky(S + alpha*np.eye(*S.shape))
            return L
        except:
            if alpha == 0:
                alpha = 1e-10
            else:
                alpha *= 10
            print(">>>> {}::cholesky: Matrix is not positive definite, adding {:.0e} on the diagonal"
                .format(os.path.basename(__file__), alpha))
