import tensorflow as tf
import numpy
import matplotlib.pyplot as plot


#####data parameters###########################
n = 3 #size of board
k = 3 #number in row to win, 
#n = 3, k = 3 is tic tac toe
#NOTE: code does not actually implement arbitrary k/n at the moment.


#####network/learning parameters###############

nhidden = 20
descriptor_output_size = (n+n+2)*(4) #(position: n rows + n columns + 2 diagonals) * (interesting state: 3 in row for me, 3 in row for him, unblocked 2 in row for me, unblocked 2 in row for him) TODO: include other useful feature descriptors, e.g. "forks"
discount_factor = 0.98
eta = 0.005
eta_decay = 0.85 #Multiplicative decay per epoch


###############################################
tf.set_random_seed(1) 
numpy.random.seed(1)



def threeinrow(state): #helper, expects state to be in square shape, only looks for positive 3 in row
    return numpy.any(numpy.sum(state,axis=0) == 3) or numpy.any(numpy.sum(state,axis=1) == 3) or numpy.sum(numpy.diagonal(state)) == 3 or  numpy.sum(numpy.diagonal(numpy.fliplr(state))) == 3 

def unblockedopptwo(state): #helper, expects state to be in square shape, only looks for negative 2 in row without positive one in remaining spot.
    return numpy.any(numpy.sum(state,axis=0) == -2) or numpy.any(numpy.sum(state,axis=1) == -2) or numpy.sum(numpy.diagonal(state)) == -2 or numpy.sum(numpy.diagonal(numpy.fliplr(state))) == -2 
    
def catsgame(state):  #helper, expects state to be in square shape, checks whether state is a cats game 
    return numpy.sum(numpy.abs(state)) >= 8 and not (threeinrow(state) or unblockedopptwo(state) or threeinrow(-state) or unblockedopptwo(-state))

def reward(state):
    state = state.reshape((3,3))
    if threeinrow(state):
	return 1.
    elif unblockedopptwo(state): 
	return -1.
    return 0.

#(position: n rows + n columns + 2 diagonals) * (interesting state: 3 in row for me, 3 in row for him, unblocked 2 in row for me, unblocked 2 in row for him) TODO: include other useful feature descriptors, e.g. "forks"
def description_target(state): #helper, generates description target for a given state
    if unblockedopptwo(state) or unblockedopptwo(-state) or threeinrow(state) or threeinrow(-state):
	target = []
	#rows
	for i in xrange(n):
	    if numpy.sum(state[i,:]) == 3:
		target.extend([1,-1,-1,-1])	
	    elif numpy.sum(state[i,:]) == -3:
		target.extend([-1,1,-1,-1])	
	    elif numpy.sum(state[i,:]) == 2:
		target.extend([-1,-1,1,-1])	
	    elif numpy.sum(state[i,:]) == -2:
		target.extend([-1,-1,-1,1])	
	    else:
		target.extend([-1,-1,-1,-1])	
	#columns
	for i in xrange(n):
	    if numpy.sum(state[:,i]) == 3:
		target.extend([1,-1,-1,-1])	
	    elif numpy.sum(state[:,i]) == -3:
		target.extend([-1,1,-1,-1])	
	    elif numpy.sum(state[:,i]) == 2:
		target.extend([-1,-1,1,-1])	
	    elif numpy.sum(state[:,i]) == -2:
		target.extend([-1,-1,-1,1])	
	    else:
		target.extend([-1,-1,-1,-1])	
	#diagonals
	diag1 = numpy.diag(state)
	diag2 = numpy.diag(numpy.fliplr(state)) 
	for d in [diag1,diag2]:
	    if numpy.sum(d) == 3:
		target.extend([1,-1,-1,-1])	
	    elif numpy.sum(d) == -3:
		target.extend([-1,1,-1,-1])	
	    elif numpy.sum(d) == 2:
		target.extend([-1,-1,1,-1])	
	    elif numpy.sum(d) == -2:
		target.extend([-1,-1,-1,1])	
	    else:
		target.extend([-1,-1,-1,-1])	
	target = numpy.array(target)
    else: #Nothing here that goes in current descriptions
	target = -numpy.ones(descriptor_output_size)
    return target
##########Opponents##############

def random_opponent(state):
    newstate = numpy.copy(state)

    selection = numpy.random.randint(0,9)
    if numpy.shape(newstate) == (n,n): #handle non-flattened arrays
	selection = numpy.unravel_index(selection,(n,n))
    while newstate[selection] != 0:
	selection = numpy.random.randint(0,9)
	if numpy.shape(newstate) == (n,n): #handle non-flattened arrays
	    selection = numpy.unravel_index(selection,(n,n))

    newstate[selection] = -1
    return newstate 


#################################


def update_state(state,selection):
    newstate = numpy.copy(state)
    if numpy.shape(newstate) == (n,n): #handle non-flattened arrays
	selection = numpy.unravel_index(selection,(n,n))
    if newstate[selection] == 0:
	newstate[selection] = 1
	return newstate 
    else:
	return [] #illegal move, array is easier to check for than -1 because python is silly sometimes




#############Q-approx network####################
initialized_stuff = {} #Dictionary to hold weights, etc., to share initilizations between network instantiations (for fair comparison)
class Q_approx:
    def __init__(self):
	self.input_ph = tf.placeholder(tf.float32, shape=[n*n,1])
	self.target_ph = tf.placeholder(tf.float32, shape=[n*n,1])
	if initialized_stuff == {}:
	    self.W1 = tf.Variable(tf.random_normal([nhidden,n*n],0,0.1)) 
	    self.b1 = tf.Variable(tf.random_normal([nhidden,1],0,0.1))
	    self.W2 = tf.Variable(tf.random_normal([nhidden,nhidden],0,0.1))
	    self.b2 = tf.Variable(tf.random_normal([nhidden,1],0,0.1))
	    self.W3 = tf.Variable(tf.random_normal([nhidden,nhidden],0,0.1))
	    self.b3 = tf.Variable(tf.random_normal([nhidden,1],0,0.1))
	    self.W4 = tf.Variable(tf.random_normal([n*n,nhidden],0,0.1))
	    self.b4 = tf.Variable(tf.random_normal([n*n,1],0,0.1))
	    self.keep_prob = tf.placeholder(tf.float32) 
	    self.eta = tf.placeholder(tf.float32) 
	    initialized_stuff['W1'] = self.W1
	    initialized_stuff['W2'] = self.W2
	    initialized_stuff['W3'] = self.W3
	    initialized_stuff['W4'] = self.W4
	    initialized_stuff['b1'] = self.b1
	    initialized_stuff['b2'] = self.b2
	    initialized_stuff['b3'] = self.b3
	    initialized_stuff['b4'] = self.b4
	else:
	    self.W1 = tf.Variable(initialized_stuff['W1'].initialized_value())
	    self.W2 = tf.Variable(initialized_stuff['W2'].initialized_value())
	    self.W3 = tf.Variable(initialized_stuff['W3'].initialized_value())
	    self.W4 = tf.Variable(initialized_stuff['W4'].initialized_value())
	    self.b1 = tf.Variable(initialized_stuff['b1'].initialized_value())
	    self.b2 = tf.Variable(initialized_stuff['b2'].initialized_value())
	    self.b3 = tf.Variable(initialized_stuff['b3'].initialized_value())
	    self.b4 = tf.Variable(initialized_stuff['b4'].initialized_value())
	self.output = tf.nn.tanh(tf.matmul(self.W4,tf.nn.tanh(tf.matmul(self.W3,tf.nn.dropout(tf.nn.tanh(tf.matmul(self.W2,tf.nn.dropout(tf.nn.tanh(tf.matmul(self.W1,self.input_ph)+self.b1),keep_prob=self.keep_prob))+self.b2),keep_prob=self.keep_prob))+self.b3))+self.b4)
	self.error = tf.square(self.output-self.target_ph)
	self.optimizer = tf.train.AdamOptimizer(self.eta)
	self.train = self.optimizer.minimize(tf.reduce_mean(self.error))
	self.sess = tf.Session()
	self.sess.run(tf.initialize_all_variables())
	self.epsilon = 0.1 #epsilon greedy
	self.curr_eta = eta

    def __del__self():
	sess.close()	 

    def Q(self,state,keep_prob=1.0): #Outputs estimated Q-value for each move in this state
	return self.sess.run(self.output,feed_dict={self.input_ph: state.reshape((9,1)),self.keep_prob: keep_prob})  

    def train_Q(self,state):
	curr = self.Q(state,keep_prob=0.5)	
	if numpy.random.rand() > self.epsilon:
	    selection = numpy.argmax(curr)
	else:
	    selection = numpy.random.randint(0,9)
	new_state = update_state(state,selection)
	if new_state == []: #illegal move
	    curr[selection] = -1
	else:
	    this_reward = reward(new_state)
	    if this_reward in [1,-1]: #if won or lost
		curr[selection] = this_reward 
	    else:
		curr[selection] = this_reward+discount_factor*max(self.Q(new_state))
	self.sess.run(self.train,feed_dict={self.input_ph: state.reshape((9,1)),self.target_ph: curr,self.keep_prob: 0.5,self.eta: self.curr_eta}) 
	return new_state

    def Q_move(self,state,train=False): #Executes a move and returns the new state. Replaces illegal moves with random legal moves
	if train:
	    new_state = self.train_Q(state)		
	else:
	    curr = self.Q(datum,keep_prob = 1.0)	
	    new_state = update_state(datum,numpy.argmax(curr))
	if new_state == []: #illegal move -- play randomly
	    new_state = numpy.copy(state)
	    selection = numpy.random.randint(0,9)
	    if numpy.shape(state) == (n,n): #handle non-flattened arrays
		selection = numpy.unravel_index(selection,(n,n))
	    while new_state[selection] != 0:
		selection = numpy.random.randint(0,9)
		if numpy.shape(new_state) == (n,n): #handle non-flattened arrays
		    selection = numpy.unravel_index(selection,(n,n))
	    new_state[selection] = 1
	    return new_state
	else: #legal move 
	    return new_state


class Q_approx_and_descriptor(Q_approx):
    def __init__(self):
	super(Q_approx_and_descriptor,self).__init__()
	if ('W3d' not in initialized_stuff.keys()):
	    
	    self.W3d = tf.Variable(tf.random_normal([n*n,nhiddendescriptor],0,0.1))
	    self.b3d = tf.Variable(tf.random_normal([n*n,1],0,0.1))
	    self.W4d = tf.Variable(tf.random_normal([descriptor_output_size,nhiddendescriptor],0,0.1))
	    self.b4d = tf.Variable(tf.random_normal([descriptor_output_size,1],0,0.1))
	    initialized_stuff['W3d'] = self.W3d
	    initialized_stuff['b3d'] = self.b3d
	    initialized_stuff['W4d'] = self.W4d
	    initialized_stuff['b4d'] = self.b4d
	else:
	    self.W3d = tf.Variable(initialized_stuff['W3d'].initialized_value())
	    self.b3d = tf.Variable(initialized_stuff['b3d'].initialized_value())
	    self.W4d = tf.Variable(initialized_stuff['W4d'].initialized_value())
	    self.b4d = tf.Variable(initialized_stuff['b4d'].initialized_value())
	self.description_output = tf.nn.tanh(tf.matmul(self.W4d,tf.nn.tanh(tf.matmul(self.W3d,tf.nn.dropout(tf.nn.tanh(tf.matmul(self.W2,tf.nn.dropout(tf.nn.tanh(tf.matmul(self.W1,self.input_ph)+self.b1),keep_prob=self.keep_prob))+self.b2),keep_prob=self.keep_prob))+self.b3d))+self.b4d)

    def describe(self,state,keep_prob=1.0): #Outputs estimated description for the current state
	return self.sess.run(self.description_output,feed_dict={self.input_ph: state.reshape((9,1)),self.keep_prob: keep_prob})  

    def train_description(self,state):
	self.sess.run(self.train,feed_dict={self.input_ph: state.reshape((9,1)),self.target_ph: description_target(state),self.keep_prob: 0.5,self.eta: self.curr_eta}) 

#####Data generation#####################

def make_ttt_array():
    """Returns a random 3 x 3 array with i +1s, j -1s, and the rest 0, where i-j is either 0 or -1  (i.e. could occur as a sequence of alternating moves with the +1 player to go next). No legality checks."""
    i = numpy.random.randint(0,5)
    if i < 4:
	j = numpy.random.randint(i,i+1) 
    else:
	j = 4 #full board is boring
    state = numpy.zeros(9)
    locations = numpy.random.permutation(9)
    state[locations[:i]] = 1
    state[locations[i:i+j]] = -1
    return state.reshape((3,3))

def generate(generator,condition,n):
    """Generates n data points satisfying condition (a boolean function) by rejection sampling from generator"""
    data = []
    while len(data) < n:
	this_point = generator() 
	if condition(this_point):
	    data.append(this_point)
    return data


##Just practice making legal moves
#data_0 = generate(make_ttt_array,lambda x:  not (threeinrow(x) or threeinrow(-x)),400)
#test_data_0 = generate(make_ttt_array,lambda x: not (threeinrow(x) or threeinrow(-x)) and not any([numpy.array_equal(x,arr) for arr in data_0]),100)
#
##Practice getting 3 in a row
#data_1 = generate(make_ttt_array,lambda x: unblockedopptwo(-x) and not (threeinrow(x) or threeinrow(-x)),400)
#test_data_1 = generate(make_ttt_array,lambda x: unblockedopptwo(-x) and not (threeinrow(x) or threeinrow(-x)) and not any([numpy.array_equal(x,arr) for arr in data_1+data_0]),100)
#
#
#c_Q_net = Q_approx()
#
#def train(data_set,epochs=1):
#    for e in xrange(epochs):
#	order = numpy.random.permutation(len(data_set))
#	for i in order:
#	    c_Q_net.train_Q(data_set[i])	
#
#def test(test_data_set,verbose=False):
#    score = 0
#    illegal = 0
#    for datum in test_data_set:
#	curr = c_Q_net.Q(datum)	
#	new_state = update_state(datum,numpy.argmax(curr))
#	if new_state == []:
#	    illegal += 1
#	elif threeinrow(new_state):
#	    score += 1
#	elif unblockedopptwo(new_state):
#	    score += -1
#	if verbose:
#	    print(datum,new_state)
#    print "avg score = %f" %(float(score)/len(test_data_set)) 
#    print "illegal move rate = %f" %(float(illegal)/len(test_data_set)) 
#


def play_game(Q_net,opponent,train=False):
    gofirst = numpy.random.randint(0,2)
    state = numpy.zeros((3,3))
    i = 0
    while not (catsgame(state) or threeinrow(state) or threeinrow(-state)):
	if i % 2 == gofirst:
	    state = opponent(state)
	else: 
	    state = Q_net.Q_move(state,train=train) 	    
	i += 1
    if threeinrow(state):
	return 1
    if threeinrow(-state) or unblockedopptwo(state):
	return -1
    return 0
	    


basic_Q_net = Q_approx()



def train_on_games(Q_net,opponent,numgames=1000):
    for game in xrange(numgames):
	play_game(Q_net,opponent,train=True)

def test_on_games(Q_net,opponent,numgames=1000):
    score = 0
    for game in xrange(numgames):
	score += play_game(Q_net,opponent,train=True)
    print "avg test score = %f" %(float(score)/numgames) 



print "Initial test:"
test_on_games(basic_Q_net,random_opponent,numgames=10000)
print "Training..."
for i in xrange(100):
    print "training epoch %i" %i
    train_on_games(basic_Q_net,random_opponent,numgames=500)
    test_on_games(basic_Q_net,random_opponent,numgames=1000)
    basic_Q_net.curr_eta *= eta_decay

print "Training done, final test:"
test_on_games(basic_Q_net,random_opponent,numgames=10000)
