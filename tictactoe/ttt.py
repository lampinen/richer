import tensorflow as tf
import numpy
import matplotlib.pyplot as plot

####Testing parameters###############

learning_rates = [0.001]
learning_rate_decays = [0.8]
pretraining_conditions = [True,False]
num_runs_per = 20


#####data parameters###########################
n = 3 #size of board
k = 3 #number in row to win, 
#n = 3, k = 3 is tic tac toe
#NOTE: code does not actually implement arbitrary k/n at the moment.



#####network/learning parameters###############
nhidden = 20
nhiddendescriptor = 20
descriptor_output_size = (n+n+2)+(4) #(position: n rows + n columns + 2 diagonals) + (interesting state: 3 in row for me, 3 in row for him, unblocked 2 in row for me, unblocked 2 in row for him) TODO: include other useful feature descriptors, e.g. "forks"
discount_factor = 0.98
#eta = 0.005
#description_eta = 0.005
#eta_decay = 0.8 #Multiplicative decay per epoch
description_eta_decay = 0.7 #Multiplicative decay per epoch
nepochs = 10

###############################################

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
    target = []
    for i in xrange(n):
	target.extend([1 if i == j else -1 for j in xrange(n+n+2)])
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
    for i in xrange(n):
	target.extend([1 if i == j-3 else -1 for j in xrange(n+n+2)])
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
    diag = [numpy.diag(state),numpy.diag(numpy.fliplr(state))]
    for i in xrange(2):
	d = diag[i]
	target.extend([1 if i == j-6 else -1 for j in xrange(n+n+2)])
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
    return target.reshape(8,descriptor_output_size)
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
class Q_approx(object):
    def __init__(self):
	global initialized_stuff
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
	self.keep_prob = tf.placeholder(tf.float32) 
	self.output = tf.nn.tanh(tf.matmul(self.W4,tf.nn.tanh(tf.matmul(self.W3,tf.nn.dropout(tf.nn.tanh(tf.matmul(self.W2,tf.nn.dropout(tf.nn.tanh(tf.matmul(self.W1,self.input_ph)+self.b1),keep_prob=self.keep_prob))+self.b2),keep_prob=self.keep_prob))+self.b3))+self.b4)
	self.error = tf.square(self.output-self.target_ph)
	self.eta = tf.placeholder(tf.float32) 
	self.optimizer = tf.train.AdamOptimizer(self.eta)
	self.train = self.optimizer.minimize(tf.reduce_sum(self.error))
	self.epsilon = 0.1 #epsilon greedy
	self.curr_eta = eta
	self.sess = None

    def set_TF_sess(self,sess):
	self.sess = sess

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
	global initialized_stuff
	super(Q_approx_and_descriptor,self).__init__()
	self.description_target_ph = tf.placeholder(tf.float32, shape=[descriptor_output_size,1])
	self.description_input_ph = tf.placeholder(tf.float32, shape=[descriptor_output_size,1])
	if ('W3d' not in initialized_stuff.keys()):
	    
	    self.W3d = tf.Variable(tf.random_normal([nhiddendescriptor,nhidden+descriptor_output_size],0,0.1))
	    self.b3d = tf.Variable(tf.random_normal([nhiddendescriptor,1],0,0.1))
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
	self.description_output = tf.nn.tanh(tf.matmul(self.W4d,tf.nn.tanh(tf.matmul(self.W3d,tf.concat(0,(tf.nn.dropout(tf.nn.tanh(tf.matmul(self.W2,tf.nn.dropout(tf.nn.tanh(tf.matmul(self.W1,self.input_ph)+self.b1),keep_prob=self.keep_prob))+self.b2),keep_prob=self.keep_prob),self.description_input_ph)))+self.b3d))+self.b4d)
	self.description_error = tf.square(self.description_output-self.description_target_ph)
	self.description_train = self.optimizer.minimize(tf.reduce_sum(self.description_error))
	self.curr_description_eta = description_eta

    def describe(self,state,keep_prob=1.0): #Outputs estimated descriptions for the current state
	this_description = []
	for j in xrange(n+n+2):
	    this_description_input = numpy.roll([1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],j).reshape((descriptor_output_size,1))
	    this_description.append(self.sess.run(self.description_output,feed_dict={self.input_ph: state.reshape((9,1)),self.description_input_ph: this_description_input,self.keep_prob: keep_prob})) 
	return 


    def get_description_error(self,state,keep_prob=1.0): #Outputs description sum across possible positions of squared error 
	state_full_description_target = description_target(state)
	SSE = numpy.zeros((descriptor_output_size,1))
	for j in xrange(n+n+2):
	    this_description_target = state_full_description_target[j].reshape((descriptor_output_size,1))
	    this_description_input = numpy.roll([1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],j).reshape((descriptor_output_size,1))
	    SSE += self.sess.run(self.description_error,feed_dict={self.input_ph: state.reshape((9,1)),self.description_input_ph: this_description_input,self.description_target_ph: this_description_target,self.keep_prob: keep_prob}) 
	return SSE 

    def train_description(self,state): #right now, only trains on describing events given locations TODO: also include picking locations for events, but need to handle multiple occurrences
	state_full_description_target = description_target(state)
	boring_list = [] #list of places where nothing is happening, a few will be sampled at the end for balance 
	for i in xrange(n+n+2):
	    this_description_target = state_full_description_target[i].reshape((descriptor_output_size,1))
	    if sum(this_description_target[-4:]) > -4:
		this_description_input = numpy.roll([1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],i).reshape((n+n+2+4,1))
		self.sess.run(self.description_train,feed_dict={self.input_ph: state.reshape((9,1)),self.description_input_ph: this_description_input,self.description_target_ph: this_description_target,self.keep_prob: 0.5,self.eta: self.curr_description_eta}) 
	    else:
		boring_list.append(i)
	    
	#balance out with training on non-events
	boring_list = numpy.random.permutation(boring_list)[:n+n+2-len(boring_list)]
	for j in boring_list:
	    this_description_target = state_full_description_target[j].reshape((descriptor_output_size,1))
	    this_description_input = numpy.roll([1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],j).reshape((n+n+2+4,1))
	    self.sess.run(self.description_train,feed_dict={self.input_ph: state.reshape((9,1)),self.description_input_ph: this_description_input,self.description_target_ph: this_description_target,self.keep_prob: 0.5,self.eta: self.curr_description_eta}) 
	    
	    

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


def play_game(Q_net,opponent,train=False,description_train=False):
    gofirst = numpy.random.randint(0,2)
    state = numpy.zeros((3,3))
    i = 0
    while not (catsgame(state) or threeinrow(state) or threeinrow(-state)):
	if i % 2 == gofirst:
	    state = opponent(state)
	else: 
	    state = Q_net.Q_move(state,train=train) 	    
	if description_train and (unblockedopptwo(state) or unblockedopptwo(-state) or threeinrow(state) or threeinrow(-state)):
	    Q_net.train_description(state)	
	i += 1
	
    if threeinrow(state):
	return 1
    if threeinrow(-state) or unblockedopptwo(state):
	return -1
    return 0

def train_on_games(Q_net,opponent,numgames=1000):
    score = 0
    for game in xrange(numgames):
	score += play_game(Q_net,opponent,train=True)
    return  (float(score)/numgames) 

def test_on_games(Q_net,opponent,numgames=1000):
    score = 0
    for game in xrange(numgames):
	score += play_game(Q_net,opponent,train=True)
    return  (float(score)/numgames) 

def train_descriptions(Q_net,data_set,epochs=1):
    for e in xrange(epochs):
	order = numpy.random.permutation(len(data_set))
	for i in order:
	    Q_net.train_description(data_set[i])	

def test_descriptions(Q_net,data_set):
    descr_MSE = 0.0
    for i in xrange(len(data_set)):
	descr_MSE += numpy.sum(Q_net.get_description_error(data_set[i])[0])	
    print "Description MSE = %f" %(descr_MSE/len(data_set)) 
    return descr_MSE/len(data_set)

    
def train_on_games_with_descriptions(Q_net,opponent,numgames=1000,numdescriptions=100):
    description_step = numgames/numdescriptions
    score = 0
    for game in xrange(numgames):
	score += play_game(Q_net,opponent,train=True,description_train=((game%description_step)==0))
    return  (float(score)/numgames) 

for pretraining_condition in pretraining_conditions:
    for eta in learning_rates:
	description_eta = eta
	for eta_decay in learning_rate_decays:
	    avg_descr_score_track = numpy.zeros(nepochs+1)
	    avg_descr_descr_MSE_track = numpy.zeros(nepochs+1)
	    avg_basic_score_track = numpy.zeros(nepochs+1)
	    for run in xrange(num_runs_per):
		print "pretrain-%s_eta-%f_eta_decay-%f_run-%i" %(str(pretraining_condition),eta,eta_decay,run)
		tf.set_random_seed(run) 
		numpy.random.seed(run)


		initialized_stuff = {} #Dictionary to hold weights, etc., to share initilizations between network instantiations (for fair comparison)


		#network initialization
		basic_Q_net = Q_approx()
		descr_Q_net = Q_approx_and_descriptor()
		sess = tf.Session()
		basic_Q_net.set_TF_sess(sess)
		descr_Q_net.set_TF_sess(sess)
		sess.run(tf.initialize_all_variables())


		if pretraining_condition:
		    #description data creation
		    descr_train_data = generate(make_ttt_array,lambda x: unblockedopptwo(x) or unblockedopptwo(-x) or threeinrow(x) or threeinrow(-x),2000)
		descr_test_data = generate(make_ttt_array,lambda x: unblockedopptwo(x) or unblockedopptwo(-x) or threeinrow(x) or threeinrow(-x),2000)



		descr_score_track = []
		descr_descr_MSE_track = []
		basic_score_track = []

#		print "Description initial test (descr_Q_net):"
#		temp = test_descriptions(descr_Q_net,descr_test_data)
#		print temp
#		descr_descr_MSE_track.append(temp)
#		print "Initial test (basic_Q_net):"
		temp = test_on_games(basic_Q_net,random_opponent,numgames=1000)
#		print temp
		basic_score_track.append(temp)
#		print "Initial test (descr_Q_net):"
		temp = test_on_games(descr_Q_net,random_opponent,numgames=1000)
#		print temp
		descr_score_track.append(temp)


		if pretraining_condition:
		    ##Description pretraining
#		    print "Pre-training descriptions..."
		    train_descriptions(descr_Q_net,descr_train_data,epochs=1)
#		    print "Description test after pre-training (descr_Q_net):"
#		    temp = test_descriptions(descr_Q_net,descr_test_data)
#		    print temp
#		    descr_descr_MSE_track.append(temp)
		else: #otherwise test without pre-training
		    pass
#		    temp = test_descriptions(descr_Q_net,descr_test_data)
#		    descr_descr_MSE_track.append(temp)

		print "Training..."
		for i in xrange(10):
		    print "training epoch %i" %i
		    train_on_games(basic_Q_net,random_opponent,numgames=500)
		    train_on_games_with_descriptions(descr_Q_net,random_opponent,numgames=500,numdescriptions=50)

		    temp = test_on_games(basic_Q_net,random_opponent,numgames=1000)
		    print "basic_Q_net average score: %f" %temp
		    basic_score_track.append(temp)
		    temp = test_on_games(descr_Q_net,random_opponent,numgames=1000)
		    print "descr_Q_net average score: %f" %temp
		    descr_score_track.append(temp)
#		    temp = test_descriptions(descr_Q_net,descr_test_data)
#		    descr_descr_MSE_track.append(temp)
		    

		    if (i%2) == 1:
			basic_Q_net.curr_eta *= eta_decay
			descr_Q_net.curr_eta *= eta_decay
			descr_Q_net.curr_description_eta *= description_eta_decay

		sess.close()
		tf.reset_default_graph()

		avg_basic_score_track += numpy.array(basic_score_track)
		avg_descr_score_track += numpy.array(descr_score_track)
#		avg_descr_descr_MSE_track += numpy.array(descr_descr_MSE_track)


		
	    numpy.savetxt('avg_descr_score_track_pretrain-%s_eta-%f_eta_decay-%f.csv'%(str(pretraining_condition),eta,eta_decay),descr_score_track,delimiter=',')
	    numpy.savetxt('avg_basic_score_track-%s_eta-%f_eta_decay-%f.csv'%(str(pretraining_condition),eta,eta_decay),basic_score_track,delimiter=',')
#	    numpy.savetxt('avg_descr_descr_MSE_track-%s_eta-%f_eta_decay-%f.csv'%(str(pretraining_condition),eta,eta_decay),descr_descr_MSE_track,delimiter=',')

