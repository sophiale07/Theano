import numpy as np
import theano.tensor as T
from theano import function

#basic
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x,y],z)

print(f(2,3))

#pretty-print the function
from theano import pp
print(pp(z))

#matrix
x = T.dmatrix('x')
y = T.dmatrix('y')
z = T.dot(x,y)
f = function([x,y],z)
print(f(np.arange(6).reshape((3,2)),
	   5*np.ones((2,3))))
