import tensorflow as tf
import numpy
#import matplotlib.pyplot as plot

####Testing parameters###############
learning_rates = [0.05]
learning_rate_decays = [0.8]
pretraining_conditions = [True]
pct_description_conditions = [0.08]
num_runs_per = 50

#lr 0.05, decay 0.7, pretrain True, replay false, epsilon = 0.2 - some success on optimal 
#lr 0.05, decay 0.7, pretrain True, replay true (gpg = 1), epsilon = 0.2 - some success on optimal 


#####data parameters###########################
n = 3 #size of board
k = 3 #number in row to win, 
#n = 3, k = 3 is tic tac toe
#NOTE: code does not actually implement arbitrary k/n at the moment.

#####network/learning parameters###############
nhidden = 100
nhiddendescriptor = 20
descriptor_output_size =(3)+ (4) +(2) #(actual items in this row/column/diagonal) + (interesting state in this location: 3 in row, unblocked 2 in row, for me, for him) +(fork for me, fork for him) TODO: include other useful feature descriptors

descriptor_input_size = (n+n+2)
discount_factor = 1.0 #for Q-learning
epsilon = 0.2 #epsilon greedy
#eta = 0.005
#description_eta = 0.0001 #NOTE:Using replay buffer forces description_eta = eta whenever training on games with descriptions (because gradients are combined from both sources)
#eta_decay = 0.8 #Multiplicative decay per epoch
description_eta_decay = 0.7 #Multiplicative decay per epoch
description_pretraining_epochs = 2
nepochs = 20
games_per_epoch = 50

#for replay buffer 
use_replay_buffer = True
games_per_gradient = 1
assert((games_per_epoch % games_per_gradient) == 0)
###############################################

def threeinrow(state): #helper, expects state to be in square shape, only looks for positive 3 in row
    return numpy.any(numpy.sum(state,axis=0) == 3) or numpy.any(numpy.sum(state,axis=1) == 3) or numpy.sum(numpy.diagonal(state)) == 3 or  numpy.sum(numpy.diagonal(numpy.fliplr(state))) == 3 

def unblockedopptwo(state): #helper, expects state to be in square shape, only looks for negative 2 in row without positive one in remaining spot.
    return numpy.any(numpy.sum(state,axis=0) == -2) or numpy.any(numpy.sum(state,axis=1) == -2) or numpy.sum(numpy.diagonal(state)) == -2 or numpy.sum(numpy.diagonal(numpy.fliplr(state))) == -2 

def oppfork(state): #helper, expects state to be in square shape, looks for fork for -1 player (i.e. two unblockedopptwo in different directions) 
    return numpy.sum(numpy.sum(state,axis=0) == -2) + numpy.sum(numpy.sum(state,axis=1) == -2) + (numpy.sum(numpy.diagonal(state)) == -2) + (numpy.sum(numpy.diagonal(numpy.fliplr(state))) == -2) >= 2
    
def catsgame(state):  #helper, expects state to be in square shape, checks whether state is a cats game 
    return numpy.sum(numpy.abs(state)) >= 8 and not (threeinrow(state) or unblockedopptwo(state) or threeinrow(-state) or unblockedopptwo(-state))

def reward(state):
    state = state.reshape((3,3))
    if threeinrow(state):
	return 1.
    elif unblockedopptwo(state): 
	return -1.
    return 0.

def description_target(state): #helper, generates description target for a given state
    target = []
    fork_state = [oppfork(-state),oppfork(state)]
    for i in xrange(n):
#	target.extend([1 if i == j else -1 for j in xrange(n+n+2)])
	target.extend(state[i,:])
	if numpy.sum(state[i,:]) == 3:
#	    target.extend([1,-1,-1,-1,-1,-1])	
	    target.extend([1,-1,1,-1])	
	elif numpy.sum(state[i,:]) == -3:
#	    target.extend([-1,-1,1,-1,-1,-1])	
	    target.extend([1,-1,-1,1])	
	elif numpy.sum(state[i,:]) == 2:
	    target.extend([-1,1,1,-1])	
#	    if oppfork(-state):
#		target.extend([1,-1]) #Include fork information if this is involved
#	    else:
#		target.extend([-1,-1])
	elif numpy.sum(state[i,:]) == -2:
	    target.extend([-1,1,-1,1])	
#	    if oppfork(state):
#		target.extend([-1,1]) #Include fork information if this is involved
#	    else:
#		target.extend([-1,-1])
	else:
#	    target.extend([-1,-1,-1,-1,-1,-1])	
	    target.extend([-1,-1,-1,-1])	
	target.extend(fork_state)
    for i in xrange(n):
#	target.extend([1 if i == j-3 else -1 for j in xrange(n+n+2)])
	target.extend(state[:,i])
	if numpy.sum(state[:,i]) == 3:
#	    target.extend([1,-1,-1,-1,-1,-1])	
	    target.extend([1,-1,1,-1])	
	elif numpy.sum(state[:,i]) == -3:
#	    target.extend([-1,-1,1,-1,-1,-1])	
	    target.extend([1,-1,-1,1])	
	elif numpy.sum(state[:,i]) == 2:
	    target.extend([-1,1,1,-1])	
#	    if oppfork(-state):
#		target.extend([1,-1]) #Include fork information if this is involved
#	    else:
#		target.extend([-1,-1])
	elif numpy.sum(state[:,i]) == -2:
	    target.extend([-1,1,-1,1])	
#	    if oppfork(state):
#		target.extend([-1,1]) #Include fork information if this is involved
#	    else:
#		target.extend([-1,-1])
	else:
#	    target.extend([-1,-1,-1,-1,-1,-1])	
	    target.extend([-1,-1,-1,-1])	
	target.extend(fork_state)
    diag = [numpy.diag(state),numpy.diag(numpy.fliplr(state))]
    for i in xrange(2):
	d = diag[i]
#	target.extend([1 if i == j-6 else -1 for j in xrange(n+n+2)])
	target.extend(d)
	if numpy.sum(d) == 3:
#	    target.extend([1,-1,-1,-1,-1,-1])	
	    target.extend([1,-1,1,-1])	
	elif numpy.sum(d) == -3:
#	    target.extend([-1,-1,1,-1,-1,-1])	
	    target.extend([1,-1,-1,1])	
	elif numpy.sum(d) == 2:
	    target.extend([-1,1,1,-1])	
#	    if oppfork(-state):
#		target.extend([1,-1]) #Include fork information if this is involved
#	    else:
#		target.extend([-1,-1]) #Include fork information if this is involved
	elif numpy.sum(d) == -2:
	    target.extend([-1,1,-1,1])	
#	    if oppfork(state):
#		target.extend([-1,1]) #Include fork information if this is involved
#	    else:
#		target.extend([-1,-1]) #Include fork information if this is involved
	else:
	    target.extend([-1,-1,-1,-1])	
	target.extend(fork_state)
    target = numpy.array(target)
    return target.reshape(8,descriptor_output_size)

def nbyn_input_to_2bynbyn_input(state):
    """Converts inputs from n x n -1/0/+1 representation to two n x n arrays, each containing the piece locations for one player"""
    return numpy.concatenate((numpy.ndarray.flatten(1*(state == -1)),numpy.ndarray.flatten(1*(state == 1))))


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

def single_move_foresight_opponent(state):
    newstate = numpy.copy(state)
    newstate = newstate.reshape((n,n))
    if unblockedopptwo(-state) or unblockedopptwo(state):
	colsum = numpy.sum(state,axis=0)
	rowsum = numpy.sum(state,axis=1) 
	d1sum = numpy.sum(numpy.diag(state))
	d2sum = numpy.sum(numpy.diag(numpy.fliplr(state)))
	if numpy.any(rowsum == -2):
	    selection = numpy.outer(rowsum == -2,state[rowsum == -2,:] == 0) 
	elif numpy.any(colsum == -2):
	    selection = numpy.outer(state[:,colsum == -2] == 0,colsum == -2) 
	elif d1sum == -2:
	    dis0 = numpy.diag(state) == 0
	    selection = numpy.outer(dis0,dis0) 
	elif d2sum == -2:
	    dis0 = numpy.diag(numpy.fliplr(state)) == 0
	    selection = numpy.fliplr(numpy.outer(dis0,dis0)) 	
	elif numpy.any(rowsum == 2):
	    selection = numpy.outer(rowsum == 2,state[rowsum == 2,:] == 0) 
	elif numpy.any(colsum == 2):
	    selection = numpy.outer(state[:,colsum == 2] == 0,colsum == 2) 
	elif d1sum == 2:
	    dis0 = numpy.diag(state) == 0
	    selection = numpy.outer(dis0,dis0) 
	elif d2sum == 2:
	    dis0 = numpy.diag(numpy.fliplr(state)) == 0
	    selection = numpy.fliplr(numpy.outer(dis0,dis0)) 
	else:
	    print "Error! Unhandled position for single_move_foresight_opponent"
	    exit(1)
    else:
	selection = numpy.random.randint(0,9)
	selection = numpy.unravel_index(selection,(n,n))
	while newstate[selection] != 0:
	    selection = numpy.random.randint(0,9)
	    selection = numpy.unravel_index(selection,(n,n))
    newstate[selection] = -1
    return newstate.reshape(numpy.shape(state)) #return in original shape 

def single_move_foresight_unpredictable_opponent(state):
    if numpy.random.randint(0,2):
	return single_move_foresight_opponent(state)
    else:
	return random_opponent(state)

def optimal_opponent(state):
    newstate = numpy.copy(state)
    colsum = numpy.sum(state,axis=0)
    rowsum = numpy.sum(state,axis=1) 
    d1sum = numpy.sum(numpy.diag(state))
    d2sum = numpy.sum(numpy.diag(numpy.fliplr(state)))
    if numpy.sum(numpy.abs(state)) == 0: #If first play
	selection = numpy.unravel_index(4,(n,n))
    elif numpy.sum(numpy.abs(state)) == 1: #If second play
	if newstate[1,1] == 0:
	    selection = (1,1)
	else:
	    selection = [(0,0),(0,2),(2,2),(2,0)][numpy.random.randint(4)] #play in random corner
    elif numpy.sum(numpy.abs(state)) == 2: #If third play
	if newstate[0,0] == 1:
	    selection =  [(0,2),(2,2),(2,0)][numpy.random.randint(3)] #play in random other corner
	elif newstate[0,2] == 1: 
	    selection =  [(0,0),(2,2),(2,0)][numpy.random.randint(3)] #play in random other corner
	elif newstate[2,0] == 1:
	    selection =  [(0,0),(2,2),(0,2)][numpy.random.randint(3)] #play in random other corner
	elif newstate[2,2] == 1:
	    selection =  [(0,0),(0,2),(2,0)][numpy.random.randint(3)] #play in random other corner
	else:
	    selection = [(0,0),(0,2),(2,2),(2,0)][numpy.random.randint(4)] #play in random corner
    elif unblockedopptwo(-state) or unblockedopptwo(state):
	if numpy.any(rowsum == -2):
	    selection = numpy.outer(rowsum == -2,state[rowsum == -2,:] == 0) 
	elif numpy.any(colsum == -2):
	    selection = numpy.outer(state[:,colsum == -2] == 0,colsum == -2) 
	elif d1sum == -2:
	    dis0 = numpy.diag(state) == 0
	    selection = numpy.outer(dis0,dis0) 
	elif d2sum == -2:
	    dis0 = numpy.diag(numpy.fliplr(state)) == 0
	    selection = numpy.fliplr(numpy.outer(dis0,dis0)) 	
	elif numpy.any(rowsum == 2):
	    selection = numpy.outer(rowsum == 2,state[rowsum == 2,:] == 0) 
	elif numpy.any(colsum == 2):
	    selection = numpy.outer(state[:,colsum == 2] == 0,colsum == 2) 
	elif d1sum == 2:
	    dis0 = numpy.diag(state) == 0
	    selection = numpy.outer(dis0,dis0) 
	elif d2sum == 2:
	    dis0 = numpy.diag(numpy.fliplr(state)) == 0
	    selection = numpy.fliplr(numpy.outer(dis0,dis0)) 
	else:
	    print "Error! Unhandled two in row position for optimal_opponent"
	    exit(1)
    elif numpy.sum(numpy.abs(state)) == 3: #If fourth play and nothing to block or win
	if newstate[1,1] == -1: #If we hold center
	    if numpy.sum(rowsum == 1) == 1 and numpy.sum(colsum == 1) == 1: #If both plays are on center edge, adjacent 
		selection = numpy.outer(rowsum == 1,colsum == 1) 
	    elif numpy.sum(rowsum == -1) == 1 and numpy.sum(colsum == -1) == 1: #If both plays are on corners, and must be opposite or would have been caught above 
		selection = [(0,1),(1,0),(2,1),(1,2)][numpy.random.randint(4)]
	    else: #opposite center edges or one center one corner  
		if (rowsum[0] == 0 and rowsum[2] == 0) or (colsum[0] == 0 and colsum[2] == 0): #opposite center edges
		    selection = [(0,0),(0,2),(2,2),(2,0)][numpy.random.randint(4)] #play in random corner
		else: #one center one corner: play between
		    selection = numpy.outer(rowsum == 1,colsum == 1)*(state == 0) 
	else: #Opponent holds center, and must have played in opposite corner from us, or we would have blocked. Take another corner and then plays will be forced from there
		selection = [x for x in [(0,0),(0,2),(2,0),(2,2)] if state[x] == 0][numpy.random.randint(2)]
    else: #If fifth play or above and nothing to block or win
	possible_plays = [x for x in [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)] if state[x] == 0]
	for possible_play in possible_plays:
	    temp_new_state = numpy.copy(newstate)
	    temp_new_state[possible_play] = -1
	    if oppfork(state): #Making a threat is best we can hope for
		selection = possible_play
		break
	else:
	    for possible_play in possible_plays:
		temp_new_state = numpy.copy(newstate)
		temp_new_state[possible_play] = -1
		if unblockedopptwo(state): #Making a threat is best we can hope for
		    selection = possible_play
		    break
	    else:
		selection = possible_plays[numpy.random.randint(len(possible_plays))]	
    newstate[selection] = -1
    return newstate.reshape(numpy.shape(state)) #return in original shape 


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
#initialized_stuff = {} #Dictionary to hold weights, etc., to share initilizations between network instantiations (for fair comparison)
class Q_approx(object):
    def __init__(self):
	global initialized_stuff
	self.input_ph = tf.placeholder(tf.float32, shape=[n*n,1])
	self.target_ph = tf.placeholder(tf.float32, shape=[n*n,1])
#	if initialized_stuff == {}:
	self.W1 = tf.Variable(tf.random_normal([nhidden,n*n],0,0.1)) 
	self.b1 = tf.Variable(tf.random_normal([nhidden,1],0,0.1))
	self.W2 = tf.Variable(tf.random_normal([nhidden,nhidden],0,0.1))
	self.b2 = tf.Variable(tf.random_normal([nhidden,1],0,0.1))
	self.W3 = tf.Variable(tf.random_normal([nhidden,nhidden],0,0.1))
	self.b3 = tf.Variable(tf.random_normal([nhidden,1],0,0.1))
	self.W4 = tf.Variable(tf.random_normal([n*n,nhidden],0,0.1))
	self.b4 = tf.Variable(tf.random_normal([n*n,1],0,0.1))
#	    initialized_stuff['W1'] = self.W1
#	    initialized_stuff['W2'] = self.W2
#	    initialized_stuff['W3'] = self.W3
#	    initialized_stuff['W4'] = self.W4
#	    initialized_stuff['b1'] = self.b1
#	    initialized_stuff['b2'] = self.b2
#	    initialized_stuff['b3'] = self.b3
#	    initialized_stuff['b4'] = self.b4
#	else:
#	    self.W1 = tf.Variable(initialized_stuff['W1'].initialized_value())
#	    self.W2 = tf.Variable(initialized_stuff['W2'].initialized_value())
#	    self.W3 = tf.Variable(initialized_stuff['W3'].initialized_value())
#	    self.W4 = tf.Variable(initialized_stuff['W4'].initialized_value())
#	    self.b1 = tf.Variable(initialized_stuff['b1'].initialized_value())
#	    self.b2 = tf.Variable(initialized_stuff['b2'].initialized_value())
#	    self.b3 = tf.Variable(initialized_stuff['b3'].initialized_value())
#	    self.b4 = tf.Variable(initialized_stuff['b4'].initialized_value())
	self.keep_prob = tf.placeholder(tf.float32) 
	self.output = tf.nn.tanh(tf.matmul(self.W4,tf.nn.tanh(tf.matmul(self.W3,tf.nn.dropout(tf.nn.tanh(tf.matmul(self.W2,tf.nn.dropout(tf.nn.tanh(tf.matmul(self.W1,self.input_ph)+self.b1),keep_prob=self.keep_prob))+self.b2),keep_prob=self.keep_prob))+self.b3))+self.b4)
	self.var_list = [self.W1,self.W2,self.W3,self.W4,self.b1,self.b2,self.b3,self.b4] #bookkeeping for gradients
	self.error = tf.square(self.output-self.target_ph)
	self.eta = tf.placeholder(tf.float32) 
	self.optimizer = tf.train.GradientDescentOptimizer(self.eta)
	self.train_gradients_and_vars = self.optimizer.compute_gradients(tf.reduce_sum(self.error))
	self.get_train_gradients = [g for (g,v) in self.train_gradients_and_vars]	
	self.get_train_variables = [v for (g,v) in self.train_gradients_and_vars]
	self.placeholder_gradients = []
	for grad_var in self.var_list:
	    self.placeholder_gradients.append((tf.placeholder('float', shape=grad_var.get_shape()) ,grad_var))
	self.apply_gradients = self.optimizer.apply_gradients(self.placeholder_gradients)
	self.train = self.optimizer.minimize(tf.reduce_sum(self.error))
	self.epsilon = epsilon #epsilon greedy
	self.curr_eta = eta
	self.sess = None

    def set_TF_sess(self,sess):
	self.sess = sess

    def Q(self,state,keep_prob=1.0): #Outputs estimated Q-value for each move in this state
	return self.sess.run(self.output,feed_dict={self.input_ph: (state).reshape((9,1)),self.keep_prob: keep_prob})  

    def train_Q(self,state,replay_buffer):
	curr = self.Q(state,keep_prob=0.5)	
	if numpy.random.rand() > self.epsilon:
	    curr_legal = numpy.copy(curr) 
	    curr_legal[numpy.reshape(state,(9,1)) != 0] = -numpy.inf #Filter out illegal moves
	    selection = numpy.argmax(curr_legal) #only selects legal moves
	else:
	    selection = numpy.random.randint(0,9)
	    while state[numpy.unravel_index(selection,(n,n))] != 0: #illegal move
		selection = numpy.random.randint(0,9)
	new_state = update_state(state,selection)
	this_reward = reward(new_state)
	if this_reward in [1,-1]: #if won or lost
	    curr[selection] = this_reward 
	else:
	    curr[selection] = this_reward+discount_factor*max(self.Q(new_state))
	if replay_buffer:
	    gradients = zip(self.sess.run(self.get_train_gradients,feed_dict={self.input_ph: (state).reshape((9,1)),self.target_ph: curr,self.keep_prob: 0.5,self.eta: self.curr_eta}),self.get_train_variables) 
	    return new_state,[gradients]
	else:
	    self.sess.run(self.train,feed_dict={self.input_ph: (state).reshape((9,1)),self.target_ph: curr,self.keep_prob: 0.5,self.eta: self.curr_eta}) 
	    return new_state

    def Q_move(self,state,train=False,replay_buffer=False): #Executes a move and returns the new state. Replaces illegal moves with random legal moves
	if train:
	    results = self.train_Q(state,replay_buffer)		
	else:
	    curr = self.Q(state,keep_prob = 1.0)	
	    curr_legal = numpy.copy(curr) 
	    curr_legal[numpy.reshape(state,(9,1)) != 0] = -numpy.inf #Filter out illegal moves
	    selection = numpy.argmax(curr_legal) #only selects legal moves
	    results = update_state(state,selection)
	return results

    def play_game(self,opponent,train=False,replay_buffer=False,display=False):
	gofirst = numpy.random.randint(0,2)
	state = numpy.zeros((3,3))
	i = 0
	gradients = []
	if display:
	    print "starting game..."
	while not (catsgame(state) or threeinrow(state) or threeinrow(-state)):
	    if i % 2 == gofirst:
		state = opponent(state)
	    else: 
		if not train or not replay_buffer:
		    state = self.Q_move(state,train=train,replay_buffer=replay_buffer)  
		else:
		    (state,these_gradients) = self.Q_move(state,train=train,replay_buffer=replay_buffer)  
		    gradients.extend(these_gradients)
	    i += 1
	    if display:
		print state
		print
	reward = 0
	if threeinrow(state):
	    reward = 1
	elif threeinrow(-state) or unblockedopptwo(state):
	    reward = -1
	if replay_buffer and train: 
	    return reward,gradients
	else:
	    return reward


    def play_anti_game(self,opponent,train=False,replay_buffer=False,display=False): 
	"""reward for losing instead of winning"""
	gofirst = numpy.random.randint(0,2)
	state = numpy.zeros((3,3))
	i = 0
	gradients = []
	if display:
	    print "starting game..."
	while not (catsgame(state) or threeinrow(state) or threeinrow(-state)):
	    if i % 2 == gofirst:
		state = opponent(state)
	    else: 
		if not train or not replay_buffer:
		    state = self.Q_move(state,train=train,replay_buffer=replay_buffer)  
		else:
		    (state,these_gradients) = self.Q_move(state,train=train,replay_buffer=replay_buffer)  
		    gradients.extend(these_gradients)
	    i += 1
	    if display:
		print state
		print
	reward = 0
	if threeinrow(state):
	    reward = -1
	elif threeinrow(-state) or unblockedopptwo(state):
	    reward = 1
	if replay_buffer and train: 
	    return reward,gradients
	else:
	    return reward


    def train_on_games(self,opponents,numgames=1000,replay_buffer=False):
	score = 0
	num_opponents = len(opponents)
	gradients = []
	for game in xrange(numgames):
	    if replay_buffer:
		(this_score,these_gradients) = self.play_game(opponents[game % num_opponents],train=True,replay_buffer=replay_buffer)
		score += this_score
		gradients.extend(these_gradients)
		if game % games_per_gradient == 0:
		    combined_gradients = combine_gradients(gradients)
		    this_feed_dict = {self.keep_prob: 0.5,self.eta: self.curr_eta}
		    for i, grad_var in enumerate(self.var_list):
			if grad_var in combined_gradients.keys():
			    this_feed_dict[self.placeholder_gradients[i][0]] = combined_gradients[grad_var]
			else:
			    this_feed_dict[self.placeholder_gradients[i][0]] = numpy.zeros(grad_var.get_shape()) #If this variable is irrelevant, no updates
		
		    self.sess.run(self.apply_gradients,feed_dict=this_feed_dict)
		    gradients = [] 
	    else:
		score += self.play_game(opponents[game % num_opponents],train=True,replay_buffer=replay_buffer)
	return  (float(score)/numgames) 

    def train_on_anti_games(self,opponents,numgames=1000,replay_buffer=False):
	score = 0
	num_opponents = len(opponents)
	gradients = []
	for game in xrange(numgames):
	    if replay_buffer:
		(this_score,these_gradients) = self.play_anti_game(opponents[game % num_opponents],train=True,replay_buffer=replay_buffer)
		score += this_score
		gradients.extend(these_gradients)
		if game % games_per_gradient == 0:
		    combined_gradients = combine_gradients(gradients)
		    this_feed_dict = {self.keep_prob: 0.5,self.eta: self.curr_eta}
		    for i, grad_var in enumerate(self.var_list):
			if grad_var in combined_gradients.keys():
			    this_feed_dict[self.placeholder_gradients[i][0]] = combined_gradients[grad_var]
			else:
			    this_feed_dict[self.placeholder_gradients[i][0]] = numpy.zeros(grad_var.get_shape()) #If this variable is irrelevant, no updates
		
		    self.sess.run(self.apply_gradients,feed_dict=this_feed_dict)
		    gradients = [] 
	    else:
		score += self.play_anti_game(opponents[game % num_opponents],train=True,replay_buffer=replay_buffer)
	return  (float(score)/numgames) 

    def test_on_games(self,opponent,numgames=1000):
	wins = 0
	draws = 0
	losses = 0
	for game in xrange(numgames):
	    this_score = self.play_game(opponent,train=False)
	    if this_score == 1:
		wins += 1
	    elif this_score == -1:
		losses += 1
	    else:
		draws += 1 
	    
	return  (float(wins)/numgames,float(draws)/numgames,float(losses)/numgames) 

    def test_on_anti_games(self,opponent,numgames=1000):
	wins = 0
	draws = 0
	losses = 0
	for game in xrange(numgames):
	    this_score = self.play_anti_game(opponent,train=False)
	    if this_score == 1:
		wins += 1
	    elif this_score == -1:
		losses += 1
	    else:
		draws += 1 
	    
	return  (float(wins)/numgames,float(draws)/numgames,float(losses)/numgames) 

class Q_approx_and_descriptor(Q_approx):
    def __init__(self):
	global initialized_stuff
	super(Q_approx_and_descriptor,self).__init__()
	self.description_target_ph = tf.placeholder(tf.float32, shape=[descriptor_output_size,1])
	self.description_input_ph = tf.placeholder(tf.float32, shape=[descriptor_input_size,1])
#	if ('W3d' not in initialized_stuff.keys()):
	self.W3d = tf.Variable(tf.random_normal([nhiddendescriptor,nhidden+descriptor_input_size],0,0.1))
	self.b3d = tf.Variable(tf.random_normal([nhiddendescriptor,1],0,0.1))
	self.W4d = tf.Variable(tf.random_normal([descriptor_output_size,nhiddendescriptor],0,0.1))
	self.b4d = tf.Variable(tf.random_normal([descriptor_output_size,1],0,0.1))
#	    initialized_stuff['W3d'] = self.W3d
#	    initialized_stuff['b3d'] = self.b3d
#	    initialized_stuff['W4d'] = self.W4d
#	    initialized_stuff['b4d'] = self.b4d
#	else:
#	    self.W3d = tf.Variable(initialized_stuff['W3d'].initialized_value())
#	    self.b3d = tf.Variable(initialized_stuff['b3d'].initialized_value())
#	    self.W4d = tf.Variable(initialized_stuff['W4d'].initialized_value())
#	    self.b4d = tf.Variable(initialized_stuff['b4d'].initialized_value())
	self.description_output = tf.nn.tanh(tf.matmul(self.W4d,tf.nn.tanh(tf.matmul(self.W3d,tf.concat(0,(tf.nn.dropout(tf.nn.tanh(tf.matmul(self.W2,tf.nn.dropout(tf.nn.tanh(tf.matmul(self.W1,self.input_ph)+self.b1),keep_prob=self.keep_prob))+self.b2),keep_prob=self.keep_prob),self.description_input_ph)))+self.b3d))+self.b4d)
	self.var_list = [self.W1,self.W2,self.W3,self.W4,self.W3d,self.W4d,self.b1,self.b2,self.b3,self.b4,self.b3d,self.b4d] #bookkeeping for gradients
	self.description_error = tf.square(self.description_output-self.description_target_ph)
	self.description_train = self.optimizer.minimize(tf.reduce_sum(self.description_error))
	self.train_gradients_and_vars = self.optimizer.compute_gradients(tf.reduce_sum(self.error),var_list=self.var_list)
	self.get_train_gradients = [g for (g,v) in self.train_gradients_and_vars if g is not None]
	self.get_train_variables = [v for (g,v) in self.train_gradients_and_vars if g is not None]
	self.description_train_gradients_and_vars = self.optimizer.compute_gradients(tf.reduce_sum(self.description_error),var_list=self.var_list)
	self.get_description_train_gradients = [g for (g,v) in self.description_train_gradients_and_vars if g is not None]
	self.get_description_train_variables = [v  for (g,v) in self.description_train_gradients_and_vars if g is not None]
	self.placeholder_gradients = []
	for grad_var in self.var_list:
	    self.placeholder_gradients.append((tf.placeholder('float', shape=grad_var.get_shape()) ,grad_var))
	self.apply_gradients = self.optimizer.apply_gradients(self.placeholder_gradients)
	self.curr_description_eta = description_eta


    def describe(self,state,keep_prob=1.0): #Outputs estimated descriptions for the current state
	this_description = []
	for j in xrange(n+n+2):
	    this_description_input = numpy.roll([1,-1,-1,-1,-1,-1,-1,-1],j).reshape((descriptor_input_size,1))
	    this_description.append(self.sess.run(self.description_output,feed_dict={self.input_ph: (state).reshape((9,1)),self.description_input_ph: this_description_input,self.keep_prob: keep_prob})) 
	return this_description 


    def get_description_error(self,state,keep_prob=1.0): #Outputs description sum across possible positions of squared error 
	state_full_description_target = description_target(state)
	SSE = numpy.zeros((descriptor_output_size,1))
	for j in xrange(n+n+2):
	    this_description_target = state_full_description_target[j].reshape((descriptor_output_size,1))
	    this_description_input = numpy.roll([1,-1,-1,-1,-1,-1,-1,-1],j).reshape((descriptor_input_size,1))
	    SSE += self.sess.run(self.description_error,feed_dict={self.input_ph: (state).reshape((9,1)),self.description_input_ph: this_description_input,self.description_target_ph: this_description_target,self.keep_prob: keep_prob}) 
	return SSE 

    def train_description(self,state,replay_buffer=False): #right now, only trains on describing events given locations TODO: also include picking locations for events, but need to handle multiple occurrences
	state_full_description_target = description_target(state)
	boring_list = [] #list of places where nothing is happening, a few will be sampled at the end for balance 
	gradients = []
	for i in xrange(n+n+2):
	    this_description_target = state_full_description_target[i].reshape((descriptor_output_size,1))
	    if sum(this_description_target[-6:]) > -6:
		this_description_input = numpy.roll([1,-1,-1,-1,-1,-1,-1,-1],i).reshape((descriptor_input_size,1))
		if replay_buffer:
		    gradients.append(zip(self.sess.run(self.get_description_train_gradients,feed_dict={self.input_ph: (state).reshape((9,1)),self.description_input_ph: this_description_input,self.description_target_ph: this_description_target,self.keep_prob: 0.5,self.eta: self.curr_description_eta}),self.get_description_train_variables)) 
		else:
		    self.sess.run(self.description_train,feed_dict={self.input_ph: (state).reshape((9,1)),self.description_input_ph: this_description_input,self.description_target_ph: this_description_target,self.keep_prob: 0.5,self.eta: self.curr_description_eta}) 
	    else:
		boring_list.append(i)
	    
	#balance out with training on non-events
	boring_list = numpy.random.permutation(boring_list)[:n+n+2-len(boring_list)]
	for j in boring_list:
	    this_description_target = state_full_description_target[j].reshape((descriptor_output_size,1))
	    this_description_input = numpy.roll([1,-1,-1,-1,-1,-1,-1,-1],j).reshape((descriptor_input_size,1))
	    if replay_buffer:
		gradients.append(zip(self.sess.run(self.get_description_train_gradients,feed_dict={self.input_ph: (state).reshape((9,1)),self.description_input_ph: this_description_input,self.description_target_ph: this_description_target,self.keep_prob: 0.5,self.eta: self.curr_description_eta}),self.get_description_train_variables))
	    else:
		self.sess.run(self.description_train,feed_dict={self.input_ph: (state).reshape((9,1)),self.description_input_ph: this_description_input,self.description_target_ph: this_description_target,self.keep_prob: 0.5,self.eta: self.curr_description_eta}) 

	return gradients

    def play_game(self,opponent,train=False,description_train=False,replay_buffer=False,display=False):
	gofirst = numpy.random.randint(0,2)
	state = numpy.zeros((3,3))
	i = 0
	gradients = []
	if display:
	    print "starting game..."
	while not (catsgame(state) or threeinrow(state) or threeinrow(-state)):
	    if i % 2 == gofirst:
		state = opponent(state)
	    else: 
		if not train or not replay_buffer:
		    state = self.Q_move(state,train=train,replay_buffer=replay_buffer)  
		else:
		    (state,these_gradients) = self.Q_move(state,train=train,replay_buffer=replay_buffer)  
		    gradients.extend(these_gradients)
	    if description_train and (unblockedopptwo(state) or unblockedopptwo(-state) or threeinrow(state) or threeinrow(-state)):
		gradients.extend(self.train_description(state,replay_buffer=replay_buffer))
	    i += 1
	    if display:
		print state
		print
	reward = 0
	if threeinrow(state):
	    reward = 1
	elif threeinrow(-state) or unblockedopptwo(state):
	    reward = -1
	if replay_buffer and train: 
	    return reward,gradients
	else:
	    return reward
	    
    def play_anti_game(self,opponent,train=False,description_train=False,replay_buffer=False,display=False):
	gofirst = numpy.random.randint(0,2)
	state = numpy.zeros((3,3))
	i = 0
	gradients = []
	if display:
	    print "starting game..."
	while not (catsgame(state) or threeinrow(state) or threeinrow(-state)):
	    if i % 2 == gofirst:
		state = opponent(state)
	    else: 
		if not train or not replay_buffer:
		    state = self.Q_move(state,train=train,replay_buffer=replay_buffer)  
		else:
		    (state,these_gradients) = self.Q_move(state,train=train,replay_buffer=replay_buffer)  
		    gradients.extend(these_gradients)
	    if description_train and (unblockedopptwo(state) or unblockedopptwo(-state) or threeinrow(state) or threeinrow(-state)):
		gradients.extend(self.train_description(state,replay_buffer=replay_buffer))
	    i += 1
	    if display:
		print state
		print
	reward = 0
	if threeinrow(state):
	    reward = -1
	elif threeinrow(-state) or unblockedopptwo(state):
	    reward = 1
	if replay_buffer and train: 
	    return reward,gradients
	else:
	    return reward

    def train_descriptions(self,data_set,epochs=1):
	for e in xrange(epochs):
	    order = numpy.random.permutation(len(data_set))
	    for i in order:
		self.train_description(data_set[i])	
#	    self.curr_description_eta *= description_eta_decay

    def test_descriptions(self,data_set):
	descr_MSE = 0.0
	for i in xrange(len(data_set)):
	    descr_MSE += numpy.sum(self.get_description_error(data_set[i]))	
	return descr_MSE/(len(data_set)*8*descriptor_output_size)

    def train_on_games_with_descriptions(self,opponents,numgames=1000,pctdescriptions=0.2,replay_buffer=False):
	if pct_descriptions == 0.0:
	    return self.train_on_games(opponents,numgames,replay_buffer)
	description_step = numgames*pctdescriptions
	score = 0
	num_opponents = len(opponents)
	gradients = []
	for game in xrange(numgames):
	    if replay_buffer:
		(this_score,these_gradients) = self.play_game(opponents[game % num_opponents],train=True,description_train=((game%description_step)==0),replay_buffer=replay_buffer)
		score += this_score
		gradients.extend(these_gradients)
		if game % games_per_gradient == 0:
		    combined_gradients = combine_gradients(gradients)
		    this_feed_dict = {self.keep_prob: 0.5,self.eta: self.curr_eta}
		    for i, grad_var in enumerate(self.var_list):
			if grad_var in combined_gradients.keys():
			    this_feed_dict[self.placeholder_gradients[i][0]] = combined_gradients[grad_var]
			else:
			    this_feed_dict[self.placeholder_gradients[i][0]] = numpy.zeros(grad_var.get_shape()) #If this variable is irrelevant, no updates
		    self.sess.run(self.apply_gradients,feed_dict=this_feed_dict)
		    gradients = [] 
	    else:
		score += self.play_game(opponents[game % num_opponents],train=True,description_train=((game%description_step)==0),replay_buffer=replay_buffer)
	return  (float(score)/numgames) 
	    

    def train_on_anti_games_with_descriptions(self,opponents,numgames=1000,pctdescriptions=0.2,replay_buffer=False):
	if pct_descriptions == 0.0:
	    return self.train_on_anti_games(opponents,numgames,replay_buffer)
	description_step = numgames*pctdescriptions
	score = 0
	num_opponents = len(opponents)
	gradients = []
	for game in xrange(numgames):
	    if replay_buffer:
		(this_score,these_gradients) = self.play_anti_game(opponents[game % num_opponents],train=True,description_train=((game%description_step)==0),replay_buffer=replay_buffer)
		score += this_score
		gradients.extend(these_gradients)
		if game % games_per_gradient == 0:
		    combined_gradients = combine_gradients(gradients)
		    this_feed_dict = {self.keep_prob: 0.5,self.eta: self.curr_eta}
		    for i, grad_var in enumerate(self.var_list):
			if grad_var in combined_gradients.keys():
			    this_feed_dict[self.placeholder_gradients[i][0]] = combined_gradients[grad_var]
			else:
			    this_feed_dict[self.placeholder_gradients[i][0]] = numpy.zeros(grad_var.get_shape()) #If this variable is irrelevant, no updates
		    self.sess.run(self.apply_gradients,feed_dict=this_feed_dict)
		    gradients = [] 
	    else:
		score += self.play_anti_game(opponents[game % num_opponents],train=True,description_train=((game%description_step)==0),replay_buffer=replay_buffer)
	return  (float(score)/numgames) 


class Q_approx_and_autoencoder(Q_approx):
    def __init__(self):
	global initialized_stuff
	super(Q_approx_and_autoencoder,self).__init__()
	autoencoding_output_size = 9
	self.autoencoder_target_ph = tf.placeholder(tf.float32, shape=[autoencoding_output_size,1])
	self.W3a = tf.Variable(tf.random_normal([nhiddendescriptor,nhidden],0,0.1))
	self.b3a = tf.Variable(tf.random_normal([nhiddendescriptor,1],0,0.1))
	self.W4a = tf.Variable(tf.random_normal([autoencoding_output_size,nhiddendescriptor],0,0.1))
	self.b4a = tf.Variable(tf.random_normal([autoencoding_output_size,1],0,0.1))
	self.autoencoder_output = tf.nn.tanh(tf.matmul(self.W4a,tf.nn.tanh(tf.matmul(self.W3a,tf.nn.dropout(tf.nn.tanh(tf.matmul(self.W2,tf.nn.dropout(tf.nn.tanh(tf.matmul(self.W1,self.input_ph)+self.b1),keep_prob=self.keep_prob))+self.b2),keep_prob=self.keep_prob))+self.b3a))+self.b4a)
	self.var_list = [self.W1,self.W2,self.W3,self.W4,self.W3a,self.W4a,self.b1,self.b2,self.b3,self.b4,self.b3a,self.b4a] #bookkeeping for gradients
	self.autoencoder_error = tf.square(self.autoencoder_output-self.autoencoder_target_ph)
	self.autoencoder_train = self.optimizer.minimize(tf.reduce_sum(self.autoencoder_error))
	self.train_gradients_and_vars = self.optimizer.compute_gradients(tf.reduce_sum(self.error),var_list=self.var_list)
	self.get_train_gradients = [g for (g,v) in self.train_gradients_and_vars if g is not None]
	self.get_train_variables = [v for (g,v) in self.train_gradients_and_vars if g is not None]
	self.autoencoder_train_gradients_and_vars = self.optimizer.compute_gradients(tf.reduce_sum(self.autoencoder_error),var_list=self.var_list)
	self.get_autoencoder_train_gradients = [g for (g,v) in self.autoencoder_train_gradients_and_vars if g is not None]
	self.get_autoencoder_train_variables = [v  for (g,v) in self.autoencoder_train_gradients_and_vars if g is not None]
	self.placeholder_gradients = []
	for grad_var in self.var_list:
	    self.placeholder_gradients.append((tf.placeholder('float', shape=grad_var.get_shape()) ,grad_var))
	self.apply_gradients = self.optimizer.apply_gradients(self.placeholder_gradients)
	self.curr_autoencoder_eta = description_eta

    def get_autoencoder_error(self,state,keep_prob=1.0): #Outputs autoencoder sum across possible positions of squared error 
	SSE = self.sess.run(self.autoencoder_error,feed_dict={self.input_ph: (state).reshape((9,1)),self.autoencoder_target_ph: state.reshape((9,1)),self.keep_prob: keep_prob}) 
	return SSE 

    def train_autoencoder(self,state,replay_buffer=False): 
	gradients = []
	if replay_buffer:
	    gradients.append(zip(self.sess.run(self.get_autoencoder_train_gradients,feed_dict={self.input_ph: (state).reshape((9,1)),self.autoencoder_target_ph: state.reshape((9,1)),self.keep_prob: 0.5,self.eta: self.curr_autoencoder_eta}),self.get_autoencoder_train_variables)) 
	else:
	    self.sess.run(self.autoencoder_train,feed_dict={self.input_ph: (state).reshape((9,1)),self.autoencoder_target_ph: state.reshape((9,1)),self.keep_prob: 0.5,self.eta: self.curr_autoencoder_eta}) 
	return gradients

    def play_game(self,opponent,train=False,autoencoder_train=False,replay_buffer=False,display=False):
	gofirst = numpy.random.randint(0,2)
	state = numpy.zeros((3,3))
	i = 0
	gradients = []
	if display:
	    print "starting game..."
	while not (catsgame(state) or threeinrow(state) or threeinrow(-state)):
	    if i % 2 == gofirst:
		state = opponent(state)
	    else: 
		if not train or not replay_buffer:
		    state = self.Q_move(state,train=train,replay_buffer=replay_buffer)  
		else:
		    (state,these_gradients) = self.Q_move(state,train=train,replay_buffer=replay_buffer)  
		    gradients.extend(these_gradients)
	    if autoencoder_train and (unblockedopptwo(state) or unblockedopptwo(-state) or threeinrow(state) or threeinrow(-state)):
		gradients.extend(self.train_autoencoder(state,replay_buffer=replay_buffer))
	    i += 1
	    if display:
		print state
		print
	reward = 0
	if threeinrow(state):
	    reward = 1
	elif threeinrow(-state) or unblockedopptwo(state):
	    reward = -1
	if replay_buffer and train: 
	    return reward,gradients
	else:
	    return reward
	    
    def play_anti_game(self,opponent,train=False,autoencoder_train=False,replay_buffer=False,display=False):
	gofirst = numpy.random.randint(0,2)
	state = numpy.zeros((3,3))
	i = 0
	gradients = []
	if display:
	    print "starting game..."
	while not (catsgame(state) or threeinrow(state) or threeinrow(-state)):
	    if i % 2 == gofirst:
		state = opponent(state)
	    else: 
		if not train or not replay_buffer:
		    state = self.Q_move(state,train=train,replay_buffer=replay_buffer)  
		else:
		    (state,these_gradients) = self.Q_move(state,train=train,replay_buffer=replay_buffer)  
		    gradients.extend(these_gradients)
	    if autoencoder_train and (unblockedopptwo(state) or unblockedopptwo(-state) or threeinrow(state) or threeinrow(-state)):
		gradients.extend(self.train_autoencoder(state,replay_buffer=replay_buffer))
	    i += 1
	    if display:
		print state
		print
	reward = 0
	if threeinrow(state):
	    reward = -1
	elif threeinrow(-state) or unblockedopptwo(state):
	    reward = 1
	if replay_buffer and train: 
	    return reward,gradients
	else:
	    return reward

    def train_autoencoding(self,data_set,epochs=1):
	for e in xrange(epochs):
	    order = numpy.random.permutation(len(data_set))
	    for i in order:
		self.train_autoencoder(data_set[i])	
#	    self.curr_autoencoder_eta *= description_eta_decay

    def test_autoencoder(self,data_set):
	autoencoder_MSE = 0.0
	for i in xrange(len(data_set)):
	    autoencoder_MSE += numpy.sum(self.get_autoencoder_error(data_set[i]))	
	return autoencoder_MSE/(len(data_set)*3*3)

    def train_on_games_with_autoencoder(self,opponents,numgames=1000,pctautoencoding=0.2,replay_buffer=False):
	if pctautoencoding == 0.0:
	    return self.train_on_games(opponents,numgames,replay_buffer)
	autoencoder_step = numgames*pctautoencoding
	score = 0
	num_opponents = len(opponents)
	gradients = []
	for game in xrange(numgames):
	    if replay_buffer:
		(this_score,these_gradients) = self.play_game(opponents[game % num_opponents],train=True,autoencoder_train=((game%autoencoder_step)==0),replay_buffer=replay_buffer)
		score += this_score
		gradients.extend(these_gradients)
		if game % games_per_gradient == 0:
		    combined_gradients = combine_gradients(gradients)
		    this_feed_dict = {self.keep_prob: 0.5,self.eta: self.curr_eta}
		    for i, grad_var in enumerate(self.var_list):
			if grad_var in combined_gradients.keys():
			    this_feed_dict[self.placeholder_gradients[i][0]] = combined_gradients[grad_var]
			else:
			    this_feed_dict[self.placeholder_gradients[i][0]] = numpy.zeros(grad_var.get_shape()) #If this variable is irrelevant, no updates
		    self.sess.run(self.apply_gradients,feed_dict=this_feed_dict)
		    gradients = [] 
	    else:
		score += self.play_game(opponents[game % num_opponents],train=True,autoencoder_train=((game%autoencoder_step)==0),replay_buffer=replay_buffer)
	return  (float(score)/numgames) 


    def train_on_anti_games_with_autoencoder(self,opponents,numgames=1000,pctautoencoding=0.2,replay_buffer=False):
	if pctautoencoding == 0.0:
	    return self.train_on_games(opponents,numgames,replay_buffer)
	autoencoder_step = numgames*pctautoencoding
	score = 0
	num_opponents = len(opponents)
	gradients = []
	for game in xrange(numgames):
	    if replay_buffer:
		(this_score,these_gradients) = self.play_anti_game(opponents[game % num_opponents],train=True,autoencoder_train=((game%autoencoder_step)==0),replay_buffer=replay_buffer)
		score += this_score
		gradients.extend(these_gradients)
		if game % games_per_gradient == 0:
		    combined_gradients = combine_gradients(gradients)
		    this_feed_dict = {self.keep_prob: 0.5,self.eta: self.curr_eta}
		    for i, grad_var in enumerate(self.var_list):
			if grad_var in combined_gradients.keys():
			    this_feed_dict[self.placeholder_gradients[i][0]] = combined_gradients[grad_var]
			else:
			    this_feed_dict[self.placeholder_gradients[i][0]] = numpy.zeros(grad_var.get_shape()) #If this variable is irrelevant, no updates
		    self.sess.run(self.apply_gradients,feed_dict=this_feed_dict)
		    gradients = [] 
	    else:
		score += self.play_anti_game(opponents[game % num_opponents],train=True,autoencoder_train=((game%autoencoder_step)==0),replay_buffer=replay_buffer)
	return  (float(score)/numgames) 


#########end net definitions#############

def combine_gradients(gradients):
    """Computes the mean of the provided gradients by variable, returns as a dict"""
    mean_gradients = {}
    for these_grads in gradients:
	for grad,var in these_grads:
	    if var in mean_gradients.keys():
		mean_gradients[var] += grad 
	    else:
		mean_gradients[var] = grad
    
    for var in mean_gradients.keys():
	mean_gradients[var] /= float(len(gradients))
    return mean_gradients

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

    

for pretraining_condition in pretraining_conditions:
    for eta in learning_rates:
	for eta_decay in learning_rate_decays:
	    for pct_descriptions in pct_description_conditions:
		avg_descr_descr_MSE_track = numpy.zeros(2*nepochs+1)
		avg_descr_opp_random_score_track = numpy.zeros((2*nepochs+1,3))
		avg_descr_opp_optimal_score_track = numpy.zeros((2*nepochs+1,3))
		avg_basic_opp_random_score_track = numpy.zeros((2*nepochs+1,3))
		avg_basic_opp_optimal_score_track = numpy.zeros((2*nepochs+1,3))
		avg_auto_opp_random_score_track = numpy.zeros((2*nepochs+1,3))
		avg_auto_opp_optimal_score_track = numpy.zeros((2*nepochs+1,3))
		for run in xrange(num_runs_per):
		    print "pretrain-%s_eta-%f_eta_decay-%f_pct_descriptions-%f_run-%i" %(str(pretraining_condition),eta,eta_decay,pct_descriptions,run)

#		    initialized_stuff = {} #Dictionary to hold weights, etc., to share initilizations between network instantiations (for fair comparison)

		    descr_descr_MSE_track = []
		    descr_opp_random_score_track = []
		    descr_opp_optimal_score_track = []
		    basic_opp_random_score_track = []
		    basic_opp_optimal_score_track = []
		    auto_opp_random_score_track = []
		    auto_opp_optimal_score_track = []
		    for currently_training_net in ["descr","autoencoder","basic"]:
			description_eta = 0.001
			
			if (currently_training_net != "descr") and ((pretraining_condition != pretraining_conditions[0] or pct_descriptions != pct_description_conditions[0])):
			    continue #If already run the same non-descr run before, continue
			tf.set_random_seed(run) 
			numpy.random.seed(run)



			#network initialization
			if currently_training_net == "descr":
			    descr_Q_net = Q_approx_and_descriptor()
			elif currently_training_net == "autoencoder":
			    auto_Q_net = Q_approx_and_autoencoder()
			else:
			    basic_Q_net = Q_approx()

			sess = tf.Session()
			if currently_training_net == "descr":
			    descr_Q_net.set_TF_sess(sess)
			elif currently_training_net == "autoencoder":
			    auto_Q_net.set_TF_sess(sess)
			else:
			    basic_Q_net.set_TF_sess(sess)
			sess.run(tf.initialize_all_variables())


			if currently_training_net == "descr":
			    if pretraining_condition:
				#description data creation
				descr_train_data = numpy.concatenate((generate(make_ttt_array,lambda x: unblockedopptwo(x) or unblockedopptwo(-x) or threeinrow(x) or threeinrow(-x),2000),generate(make_ttt_array,lambda x: oppfork(x) or oppfork(-x),1000) )) 
			    descr_test_data = numpy.concatenate((generate(make_ttt_array,lambda x: unblockedopptwo(x) or unblockedopptwo(-x) or threeinrow(x) or threeinrow(-x),2000), generate(make_ttt_array,lambda x: oppfork(x) or oppfork(-x),1000) ))
			
			    print "Description initial test (descr_Q_net):"
			    temp = descr_Q_net.test_descriptions(descr_test_data)
			    print temp
			    #descr_descr_MSE_track.append(temp)
			    print "Initial test (descr_Q_net, random):"
			    temp = descr_Q_net.test_on_games(random_opponent,numgames=1000)
			    print temp
			    descr_opp_random_score_track.append(temp)
			    print "Initial test (descr_Q_net, optimal):"
			    temp = descr_Q_net.test_on_games(optimal_opponent,numgames=1000)
			    print temp
			    descr_opp_optimal_score_track.append(temp)
			elif currently_training_net == "autoencoder":
			    if pretraining_condition:
				auto_train_data = numpy.concatenate((generate(make_ttt_array,lambda x: unblockedopptwo(x) or unblockedopptwo(-x) or threeinrow(x) or threeinrow(-x),2000),generate(make_ttt_array,lambda x: oppfork(x) or oppfork(-x),1000) )) 
			    auto_test_data = numpy.concatenate((generate(make_ttt_array,lambda x: unblockedopptwo(x) or unblockedopptwo(-x) or threeinrow(x) or threeinrow(-x),2000), generate(make_ttt_array,lambda x: oppfork(x) or oppfork(-x),1000) ))

			    print "Description initial test (auto_Q_net):"
			    temp = auto_Q_net.test_autoencoder(auto_test_data)
			    print temp
			    #auto_auto_MSE_track.append(temp)
			    print "Initial test (auto_Q_net, random):"
			    temp = auto_Q_net.test_on_games(random_opponent,numgames=1000)
			    print temp
			    auto_opp_random_score_track.append(temp)
			    print "Initial test (auto_Q_net, optimal):"
			    temp = auto_Q_net.test_on_games(optimal_opponent,numgames=1000)
			    print temp
			    auto_opp_optimal_score_track.append(temp)

			else:
			    print "Initial test (basic_Q_net, random opponent):"
			    temp = basic_Q_net.test_on_games(random_opponent,numgames=1000)
			    print temp
			    basic_opp_random_score_track.append(temp)

			    print "Initial test (basic_Q_net, optimal opponent):"
			    temp = basic_Q_net.test_on_games(optimal_opponent,numgames=1000)
			    print temp
			    basic_opp_optimal_score_track.append(temp)

			if currently_training_net == "descr":
			    if pretraining_condition:
				##Description pretraining
				descr_Q_net.train_descriptions(descr_train_data,epochs=description_pretraining_epochs)
				print "Description test after pre-training (descr_Q_net):"
				temp = descr_Q_net.test_descriptions(descr_test_data)
				print temp
#				this_state = numpy.array([[1,0,-1],[0,1,-1],[1,0,0]])
#				print this_state
#				print descr_Q_net.describe(this_state)

				
			elif currently_training_net == "autoencoder":
			    if pretraining_condition:
				##Autoencoder pretraining
				auto_Q_net.train_autoencoding(auto_train_data,epochs=description_pretraining_epochs)
				print "Description test after pre-training (auto_Q_net):"
				temp = auto_Q_net.test_autoencoder(auto_test_data)
				print temp

			print "Training..."
			for i in xrange(2*nepochs):
			    print "training epoch %i" %i

			    if i == nepochs:
				if currently_training_net == "basic":
				    basic_Q_net.curr_eta = eta
				elif currently_training_net == "autoencoder":
				    auto_Q_net.curr_eta = eta
				    auto_Q_net.curr_autoencoder_eta = description_eta
				else:
				    descr_Q_net.curr_eta = eta
				    descr_Q_net.curr_description_eta = description_eta
				
			    
			    if i < nepochs:
				if currently_training_net == "descr": 
				    descr_Q_net.train_on_games_with_descriptions([optimal_opponent],numgames=games_per_epoch,pctdescriptions=pct_descriptions,replay_buffer=use_replay_buffer)
				elif currently_training_net == "autoencoder":
				    auto_Q_net.train_on_games_with_autoencoder([optimal_opponent],numgames=games_per_epoch,pctautoencoding=pct_descriptions,replay_buffer=use_replay_buffer)
				else:
				    basic_Q_net.train_on_games([optimal_opponent],numgames=games_per_epoch,replay_buffer=use_replay_buffer)

				if currently_training_net == "descr":
				    temp = descr_Q_net.test_on_games(random_opponent,numgames=1000)
				    print "descr_Q_net random opponent average w/d/l:",temp
				    descr_opp_random_score_track.append(temp)
				    temp = descr_Q_net.test_on_games(optimal_opponent,numgames=1000)
				    print "descr_Q_net optimal opponent average w/d/l:",temp
				    descr_opp_optimal_score_track.append(temp)
				elif currently_training_net == "autoencoder":
				    temp = auto_Q_net.test_on_games(random_opponent,numgames=1000)
				    print "auto_Q_net random opponent average w/d/l:",temp
				    auto_opp_random_score_track.append(temp)
				    temp = auto_Q_net.test_on_games(optimal_opponent,numgames=1000)
				    print "auto_Q_net optimal opponent average w/d/l:",temp
				    auto_opp_optimal_score_track.append(temp)

				else:
				    temp = basic_Q_net.test_on_games(random_opponent,numgames=1000)
				    print "basic_Q_net random opponent average w/d/l:",temp
				    basic_opp_random_score_track.append(temp)
				    temp = basic_Q_net.test_on_games(optimal_opponent,numgames=1000)
				    print "basic_Q_net optimal opponent average w/d/l:",temp
				    basic_opp_optimal_score_track.append(temp)
			    else: 
				if currently_training_net == "descr": 
				    descr_Q_net.train_on_anti_games_with_descriptions([random_opponent],numgames=games_per_epoch,pctdescriptions=pct_descriptions,replay_buffer=use_replay_buffer)
				elif currently_training_net == "autoencoder":
				    auto_Q_net.train_on_anti_games_with_autoencoder([random_opponent],numgames=games_per_epoch,pctautoencoding=pct_descriptions,replay_buffer=use_replay_buffer)
				else:
				    basic_Q_net.train_on_anti_games([random_opponent],numgames=games_per_epoch,replay_buffer=use_replay_buffer)

				if currently_training_net == "descr":
				    temp = descr_Q_net.test_on_anti_games(random_opponent,numgames=1000)
				    print "descr_Q_net random opponent average w/d/l:",temp
				    descr_opp_random_score_track.append(temp)
				    temp = descr_Q_net.test_on_anti_games(optimal_opponent,numgames=1000)
				    print "descr_Q_net optimal opponent average w/d/l:",temp
				    descr_opp_optimal_score_track.append(temp)
				elif currently_training_net == "autoencoder":
				    temp = auto_Q_net.test_on_anti_games(random_opponent,numgames=1000)
				    print "auto_Q_net random opponent average w/d/l:",temp
				    auto_opp_random_score_track.append(temp)
				    temp = auto_Q_net.test_on_anti_games(optimal_opponent,numgames=1000)
				    print "auto_Q_net optimal opponent average w/d/l:",temp
				    auto_opp_optimal_score_track.append(temp)

				else:
				    temp = basic_Q_net.test_on_anti_games(random_opponent,numgames=1000)
				    print "basic_Q_net random opponent average w/d/l:",temp
				    basic_opp_random_score_track.append(temp)
				    temp = basic_Q_net.test_on_anti_games(optimal_opponent,numgames=1000)
				    print "basic_Q_net optimal opponent average w/d/l:",temp
				    basic_opp_optimal_score_track.append(temp)


			    if (i%2) == 1:
				if currently_training_net == "basic":
				    basic_Q_net.curr_eta *= eta_decay
				elif currently_training_net == "autoencoder":
				    auto_Q_net.curr_eta *= eta_decay
				    auto_Q_net.curr_autoencoder_eta *= description_eta_decay
				else:
				    descr_Q_net.curr_eta *= eta_decay
				    descr_Q_net.curr_description_eta *= description_eta_decay
			if currently_training_net == "descr":
			    print "Description test after training (descr_Q_net):"
			    temp = descr_Q_net.test_descriptions(descr_test_data)
			    print temp
#				this_state = numpy.array([[1,0,-1],[0,1,-1],[1,0,0]])
#				print this_state
#				print descr_Q_net.describe(this_state)

				
			elif currently_training_net == "autoencoder":
			    print "Description test after training (auto_Q_net):"
			    temp = auto_Q_net.test_autoencoder(auto_test_data)
			    print temp


			sess.close()
			tf.reset_default_graph()

		    avg_descr_opp_random_score_track += numpy.array(descr_opp_random_score_track)
		    avg_descr_opp_optimal_score_track += numpy.array(descr_opp_optimal_score_track)
		    if not ((pretraining_condition != pretraining_conditions[0] or pct_descriptions != pct_description_conditions[0])):
			avg_basic_opp_optimal_score_track += numpy.array(basic_opp_optimal_score_track)
			avg_basic_opp_random_score_track += numpy.array(basic_opp_random_score_track)
			numpy.savetxt('basic_opp_random_score_track_pretrain-%s_eta-%f_eta_decay-%f_pct_descriptions-%f_run-%i.csv'%(str(pretraining_condition),eta,eta_decay,pct_descriptions,run),basic_opp_random_score_track,delimiter=',')
			numpy.savetxt('basic_opp_optimal_score_track_pretrain-%s_eta-%f_eta_decay-%f_pct_descriptions-%f_run-%i.csv'%(str(pretraining_condition),eta,eta_decay,pct_descriptions,run),basic_opp_optimal_score_track,delimiter=',')
			avg_auto_opp_optimal_score_track += numpy.array(auto_opp_optimal_score_track)
			avg_auto_opp_random_score_track += numpy.array(auto_opp_random_score_track)
			numpy.savetxt('auto_opp_random_score_track_pretrain-%s_eta-%f_eta_decay-%f_pct_descriptions-%f_run-%i.csv'%(str(pretraining_condition),eta,eta_decay,pct_descriptions,run),auto_opp_random_score_track,delimiter=',')
			numpy.savetxt('auto_opp_optimal_score_track_pretrain-%s_eta-%f_eta_decay-%f_pct_descriptions-%f_run-%i.csv'%(str(pretraining_condition),eta,eta_decay,pct_descriptions,run),auto_opp_optimal_score_track,delimiter=',')

		    numpy.savetxt('descr_opp_random_score_track_pretrain-%s_eta-%f_eta_decay-%f_pct_descriptions-%f_run-%i.csv'%(str(pretraining_condition),eta,eta_decay,pct_descriptions,run),descr_opp_random_score_track,delimiter=',')
		    numpy.savetxt('descr_opp_optimal_score_track_pretrain-%s_eta-%f_eta_decay-%f_pct_descriptions-%f_run-%i.csv'%(str(pretraining_condition),eta,eta_decay,pct_descriptions,run),descr_opp_optimal_score_track,delimiter=',')

		    
		if not ((pretraining_condition != pretraining_conditions[0] or pct_descriptions != pct_description_conditions[0])):
		    avg_basic_opp_random_score_track = avg_basic_opp_random_score_track/num_runs_per
		    numpy.savetxt('avg_basic_opp_random_score_track_pretrain-%s_eta-%f_eta_decay-%f_pct_descriptions-%f.csv'%(str(pretraining_condition),eta,eta_decay,pct_descriptions),avg_basic_opp_random_score_track,delimiter=',')
		    avg_basic_opp_optimal_score_track = avg_basic_opp_optimal_score_track/num_runs_per
		    numpy.savetxt('avg_basic_opp_optimal_score_track_pretrain-%s_eta-%f_eta_decay-%f_pct_descriptions-%f.csv'%(str(pretraining_condition),eta,eta_decay,pct_descriptions),avg_basic_opp_optimal_score_track,delimiter=',')
		    avg_auto_opp_random_score_track = avg_auto_opp_random_score_track/num_runs_per
		    numpy.savetxt('avg_auto_opp_random_score_track_pretrain-%s_eta-%f_eta_decay-%f_pct_descriptions-%f.csv'%(str(pretraining_condition),eta,eta_decay,pct_descriptions),avg_auto_opp_random_score_track,delimiter=',')
		    avg_auto_opp_optimal_score_track = avg_auto_opp_optimal_score_track/num_runs_per
		    numpy.savetxt('avg_auto_opp_optimal_score_track_pretrain-%s_eta-%f_eta_decay-%f_pct_descriptions-%f.csv'%(str(pretraining_condition),eta,eta_decay,pct_descriptions),avg_auto_opp_optimal_score_track,delimiter=',')


		avg_descr_opp_random_score_track = avg_descr_opp_random_score_track/num_runs_per
		numpy.savetxt('avg_descr_opp_random_score_track_pretrain-%s_eta-%f_eta_decay-%f_pct_descriptions-%f.csv'%(str(pretraining_condition),eta,eta_decay,pct_descriptions),avg_descr_opp_random_score_track,delimiter=',')
		avg_descr_opp_optimal_score_track = avg_descr_opp_optimal_score_track/num_runs_per
		numpy.savetxt('avg_descr_opp_optimal_score_track_pretrain-%s_eta-%f_eta_decay-%f_pct_descriptions-%f.csv'%(str(pretraining_condition),eta,eta_decay,pct_descriptions),avg_descr_opp_optimal_score_track,delimiter=',')
    #	    numpy.savetxt('avg_descr_descr_MSE_track-%s_eta-%f_eta_decay-%f.csv'%(str(pretraining_condition),eta,eta_decay),descr_descr_MSE_track,delimiter=',')

