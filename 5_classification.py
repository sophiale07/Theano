import numpy as np
import theano
import theano.tensor as T
import pickle


def compute_accuracy(y_target, y_predict):
    correct_prediction = np.equal(y_predict, y_target)
    accuracy = np.sum(correct_prediction)/len(correct_prediction)
    return accuracy

rng = np.random

N = 400                                   # training sample size
feats = 784                               # number of input variables

# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))

# Declare Theano symbolic variables
x = T.dmatrix('x')
y = T.dvector('y')

# initialize the weights and biases
W = theano.shared(rng.randn(feats),name = 'w')
b = theano.shared(0., name = 'b')

# Construct Theano expression graph
p_1 = T.nnet.sigmoid(T.dot(x,W)+b)
prediction = p_1 > 0.5
xent = -y*T.log(p_1)-(1-y)*T.log(1-p_1)  # cost entropy
cost = xent.mean() + 0.01*(W**2).sum()   #over-fitting
gW,gb = T.grad(cost,[W,b])

# Compile
learning_rate = 0.1
train = theano.function(
	inputs=[x,y],
	outputs=[prediction,xent.mean()],
	updates=((W,W-learning_rate*gW),
			 (b,b-learning_rate*gb))
)
predict = theano.function(inputs=[x],outputs=prediction)

'''
# Training
for i in range(500):
    pred,err = train(D[0],D[1])
    if i % 50 == 0:
		print("cost:",err)
		print("accuracy:",compute_accuracy(D[1], predict(D[0])))
		

print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))

# save model
with open('save/model.picke','wb') as file:
	model = [W.get_value(),b.get_value()]
	pickle.dump(model,file)
	print(model[0][:10])
	print('accuracy:',compute_accuracy(D[1],predict(D[0])))
'''
	

# load model

with open('save/model.picke','rb') as file:
	model = pickle.load(file)
	W.set_value(model[0])
	b.set_value(model[1])
	print(W.get_value()[:10])
	print('accuracy:',compute_accuracy(D[1],predict(D[0])))
	








