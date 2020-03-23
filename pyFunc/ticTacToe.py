

# vectorization / matrix decomposition functions

class ticTacToe():
    tic = self.tic
    tac = self.tac
    toe = self.toe




# ***** SCIPY Docs for Sparse Matrix /NPArray methods:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html

scipy.sparse.csc_matrix
coo_matrix
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
>>> # Constructing an empty matrix
from scipy.sparse import coo_matrix
coo_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]], dtype=int8)

>>> # Constructing a matrix using ijv format
>>> row  = np.array([0, 3, 1, 0])
>>> col  = np.array([0, 3, 1, 2])
>>> data = np.array([4, 5, 7, 9])
>>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
array([[4, 0, 9, 0],
       [0, 7, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 5]])

argmax(self[, axis, out])
Return indices of maximum elements along an axis.

argmin(self[, axis, out])
Return indices of minimum elements along an axis.

conj(self[, copy])
Element-wise complex conjugation.

conjugate(self[, copy])
Element-wise complex conjugation.

copy(self)
Returns a copy of this matrix.


diagonal(self[, k])
Returns the k-th diagonal of the matrix.

dot(self, other)
Ordinary dot product



floor(self)
Element-wise floor.

getH(self)
Return the Hermitian transpose of this matrix.

get_shape(self)
Get shape of a matrix.

getcol(self, i)
Returns a copy of column i of the matrix, as a (m x 1) CSC matrix (column vector).


todok(self[, copy])
Convert this matrix to Dictionary Of Keys format.

tolil(self[, copy])
Convert this matrix to List of Lists format.

transpose(self[, axes, copy])
Reverses the dimensions of the sparse matrix.

mean(self[, axis, dtype, out])
Compute the arithmetic mean along the specified axis.
arcsin(self)

Element-wise arcsin.

arcsinh(self)

Element-wise arcsinh.

arctan(self)

Element-wise arctan.

arctanh(self)

Element-wise arctanh.

log1p(self)

Element-wise log1p.

####
FunctionTransform(func, fromsys, tosys[, …])
A coordinate transformation defined by a function that accepts a coordinate object 
and returns the transformed coordinate object.

FunctionTransformWithFiniteDifference(func, …)
A coordinate transformation that works like a FunctionTransform, but computes velocity shifts 
based on the finite-difference relative to one of the frame attributes.
####
FunctionTransform(func, fromsys, tosys[, …])
A coordinate transformation defined by a function that accepts a coordinate object 
and returns the transformed coordinate object.

FunctionTransformWithFiniteDifference(func, …)
A coordinate transformation that works like a Function


# Reducing noise with a butter worth filter of 3rd order
# https://en.wikipedia.org/wiki/Butterworth_filter

  def __getattr__(self, attr):
        if attr == 'A':
            return self.toarray()
        elif attr == 'T':
            return self.transpose()
        elif attr == 'H':
            return self.getH()
        elif attr == 'real':
            return self._real()
        elif attr == 'imag':
            return self._imag()
        elif attr == 'size':
            return self.getnnz()
        else:
            raise AttributeError(attr + " not found")

# numpy.matrix.getH
>>> x = np.matrix(np.arange(12).reshape((3,4)))
>>> z = x - 1j*x; z
matrix([[  0. +0.j,   1. -1.j,   2. -2.j,   3. -3.j],
        [  4. -4.j,   5. -5.j,   6. -6.j,   7. -7.j],
        [  8. -8.j,   9. -9.j,  10.-10.j,  11.-11.j]])
>>> z.getH()
matrix([[ 0. -0.j,  4. +4.j,  8. +8.j],
        [ 1. +1.j,  5. +5.j,  9. +9.j],
        [ 2. +2.j,  6. +6.j, 10.+10.j],
        [ 3. +3.j,  7. +7.j, 11.+11.j]])

# np.dot
dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

# numpy.matrix.getA1
# Return self as a flattened ndarray.
# Equivalent to np.asarray(x).ravel()
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.getA1()
array([ 0,  1,  2, ...,  9, 10, 11])

>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.getA()
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

>>> m = np.matrix([[1,2], [3,4]])
>>> m.flatten()
matrix([[1, 2, 3, 4]])
>>> m.flatten('F')
matrix([[1, 3, 2, 4]])

matrix.dump(file)
Dump a pickle of the array to the specified file. The array can be read back with pickle.load or numpy.load.

numpy.matrix.dumps
matrix.dumps()
Returns the pickle of the array as a string. pickle.loads or numpy.loads will convert the string back to an array.

# import numpy as np
# print('Matrix)
# u.dot(np.diag(s).dot(vt))

# print('Rounded matrix')
# np.round(u.dot(np.diag(s).dot(vt)))




# BELOW ARE JUST NOTES FROM LABS 

# Gradient Descent with Momentum
# Compute an exponentially weighthed average of the gradients and use that gradient instead. 
# The intuitive interpretation is that this will successively dampen oscillations, improving convergence.

# Momentum:

# compute  𝑑𝑊  and  𝑑𝑏  on the current minibatch

# compute  𝑉𝑑𝑤=𝛽𝑉𝑑𝑤+(1−𝛽)𝑑𝑊  and

# compute  𝑉𝑑𝑏=𝛽𝑉𝑑𝑏+(1−𝛽)𝑑𝑏 
# These are the moving averages for the derivatives of  𝑊  and  𝑏 

# 𝑊:=𝑊−𝛼𝑉𝑑𝑤 
# 𝑏:=𝑏−𝛼𝑉𝑑𝑏 
# This averages out gradient descent, and will "dampen" oscillations. Generally,  𝛽=0.9  is a good hyperparameter value.

# # singular value decomposition
# from scipy.sparse import csc_matrix
# from scipy.sparse.linalg import svds

# # Create a sparse matrix 
# A = csc_matrix([[1, 0, 0], [5, 0, 2], [0, 1, 0], [0, 0, 3], [4, 0, 9]], dtype=float)

# # Apply SVD
# u, s, vt = svds(A, k=2) # k is the number of stretching factors

# print ('A:\n', A.toarray())
# print ('=')
# print ('\nU:\n', u)
# print ('\nΣ:\n', s)
# print ('\nV.T:\n', vt)

# """
# In this example, consider  𝐴  as the utility matrix with users and products links.

# After the decomposition  𝑈  will be the user features matrix,  
# Σ  will be the diagonal matrix of singular values (essentially weights), 
# and  𝑉.𝑇  will be the movie features matrix.

# 𝑈  and  𝑉.𝑇  are orthogonal, and represent different things.  
# 𝑈  represents how much users like each feature and  
# 𝑉.𝑇  represents how relevant each feature is to each movie.
# """

# # Now we can recreate the original ratings matrix by multiplying the three 
# # factors of the matrix together. Let's look at the exact values and then 
# # the rounded values to get an idea of what our ratings should be.


# import numpy as np
# print('Approximation of Ratings Matrix')
# u.dot(np.diag(s).dot(vt))

# print('Rounded Approximation of Ratings Matrix')
# np.round(u.dot(np.diag(s).dot(vt)))

####
FunctionTransform(func, fromsys, tosys[, …])
A coordinate transformation defined by a function that accepts a coordinate object 
and returns the transformed coordinate object.

FunctionTransformWithFiniteDifference(func, …)
A coordinate transformation that works like a FunctionTransform, but computes velocity shifts 
based on the finite-difference relative to one of the frame attributes.
####
FunctionTransform(func, fromsys, tosys[, …])
A coordinate transformation defined by a function that accepts a coordinate object 
and returns the transformed coordinate object.

FunctionTransformWithFiniteDifference(func, …)
A coordinate transformation that works like a Function
