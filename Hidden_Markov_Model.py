"""
An implementation of Hidden-Markov-Model,including evaluating、learning and decoding algorithm
@date: 2020.5.17
@author: Tingyu Mo
"""

import numpy as np

class HMM():
	def __init__(self,stateSet,obsSet,statePrior):
		#initialize parameters 
		self.stateSet = stateSet
		self.statePrior = statePrior
		self.stateNum = len(stateSet)
		self.obsnSet = obsSet
		self.obsNum = len(stateSet)
		self.stateTransMatrix = np.zeros((self.stateNum,self.stateNum))
		self.stateToObsMatrix = np.zeros((self.stateNum,self.obsNum))
		# self.lambda_=(stateTransMatrix,stateToObsMatrix,statePrior) #(A,B,pi)
		self.lambda_ = None

	def forward(self,lambda_,obs,method = "forward"):
		obsProb = 0
		
		# assert(method != "direct","direct caculate via exaustion is unfeasible!")
		if method == "forward":
			#loop caculation
			# forwardProb = np.zeros(self.stateNum)
			# for t in range(len(obs)):
			# 	print("t == ",t)
			# 	for i,s in enumerate(self.stateSet):
			# 		if t == 0:
			# 			forwardProb[i] = self.statePrior[s]*self.stateToObsMatrix[s][obs[t]]
			# 			print("state == {}, forwardProb == {}".format(s,forwardProb[i]))
			# 		else:
			# 			sumProb = 0
			# 			for s_ in stateSet:
			# 				sumProb += forwardProb[s_]*stateTransMatrix[s_][s]
			# 			print(sumProb,stateToObsMatrix[s][obs[t]])
			# 			forwardProb[i] = sumProb*self.stateToObsMatrix[s][obs[t]]
			# 			print("state == {}, forwardProb == {}".format(s,forwardProb[s]))
			
			#matrix caculation
			forwardProb = np.ones(self.stateNum)
			for t in range(len(obs)):
				if t == 0:
					forwardProb *= self.statePrior*stateToObsMatrix[:][:,obs[t]]
					# for s in self.stateSet:
					# 	print("state == {}, forwardProb == {}".format(s,forwardProb[s]))
				else:
					forwardProb = np.dot(forwardProb,stateTransMatrix)
					forwardProb *= stateToObsMatrix[:][:,obs[t]]
					# for s in self.stateSet:
					# 	print("state == {}, forwardProb == {}".format(s,forwardProb[s]))
			obsProb = sum(forwardProb)

		elif method == "backward":
			backwardProb = np.ones(self.stateNum)
			for t in range(len(obs)):
				t = len(obs) - t -1
				print("t == ",t)
				backwardProb = stateToObsMatrix[:][:,obs[t]]*np.dot(stateTransMatrix,backwardProb)
				for s in self.stateSet:
					print("state == {}, backwardProb == {}".format(s,backwardProb[s]))
			obsProb = sum(self.statePrior*backwardProb)

		elif method == "forbackward":
			timesplit = len(obs)//2
			print("forward caculation!")
			forwardProb = np.zeros(self.stateNum)
			for t in range(timesplit):
				print("t == ",t)
				if t == 0:
					forwardProb = self.statePrior*stateToObsMatrix[:][:,obs[t]]
					for s in self.stateSet:
						print("state == {}, forwardProb == {}".format(s,forwardProb[s]))
				else:
					forwardProb = np.dot(forwardProb,stateTransMatrix)
					forwardProb *= stateToObsMatrix[:][:,obs[t]]
					for s in self.stateSet:
						print("state == {}, forwardProb == {}".format(s,forwardProb[s]))
			print("backward caculation!")
			backwardProb = np.ones(self.stateNum)
			for t in range(timesplit,len(obs)):
				t = len(obs) - t -1 + timesplit
				print("t == ",t)
				backwardProb = stateToObsMatrix[:][:,obs[t]]*np.dot(stateTransMatrix,backwardProb)
				for s in self.stateSet:
					print("state == {}, backwardProb == {}".format(s,backwardProb[s]))
			obsProb = sum(np.dot(forwardProb,stateTransMatrix)*backwardProb)
		return obsProb 

	def predict(self,lambda_,obs,method ="approximate"):
		state_list = list()
		if method == "approximate":
			stateProb_list = map(self.forbackward,[t for t in range(len(obs))])
			state_list = map(self.decode,[stateProb for stateProb in stateProb_list])

		elif method == "viterbe":
			print("dynamic programming!")
			# forwardProb = np.ones(self.stateNum)
			# print("t == ",0)
			forwardProb = self.statePrior*self.stateToObsMatrix[:][:,obs[0]]
			for t in range(0,len(obs)):
				print("t == ",t)
				compete_matrix = ((forwardProb*self.stateTransMatrix.T).T)*self.stateToObsMatrix[:][:,obs[t]]
				print(list(forwardProb))
				mid_index = np.argmax(compete_matrix )
				state = self.stateSet[mid_index//3]
				forwardProb = compete_matrix[mid_index//3]
				max_index = mid_index%3
				deltaProb = forwardProb[max_index]
				state_list.append(state)
				print("max index:{} deltaProb:{}".format(max_index,deltaProb))



		return state_list

	def forbackward(self,timesplit):
		# timesplit = len(obs)//2
		print("forward caculation!")
		forwardProb = np.zeros(self.stateNum)
		for t in range(timesplit):
			print("t == ",t)
			if t == 0:
				forwardProb = self.statePrior*self.stateToObsMatrix[:][:,obs[t]]
				for s in self.stateSet:
					print("state == {}, forwardProb == {}".format(s,forwardProb[s]))
			else:
				forwardProb = np.dot(forwardProb,self.stateTransMatrix)
				forwardProb *= self.stateToObsMatrix[:][:,obs[t]]
				for s in self.stateSet:
					print("state == {}, forwardProb == {}".format(s,forwardProb[s]))
		print("backward caculation!")
		backwardProb = np.ones(self.stateNum)
		for t in range(timesplit,len(obs)):
			t = len(obs) - t -1 + timesplit
			print("t == ",t)
			backwardProb = self.stateToObsMatrix[:][:,obs[t]]*np.dot(self.stateTransMatrix,backwardProb)
			for s in self.stateSet:
				print("state == {}, backwardProb == {}".format(s,backwardProb[s]))
		stateProb =np.dot(forwardProb,self.stateTransMatrix)*backwardProb
		# obsProb = sum(np.dot(forwardProb,self.stateTransMatrix)*backwardProb)
		return stateProb

	def decode(self,stateProb):
		stateProb /= sum(stateProb)
		return self.stateSet[np.argmax(stateProb)]

if __name__ == '__main__':
	stateSet = np.array([0,1,2])
	obsSet = np.array([0,1])
	statePrior = np.array([0.2,0.4,0.4])
	stateTransMatrix = np.array([
						[0.5,0.2,0.3],
						[0.3,0.5,0.2],
						[0.2,0.3,0.5],
							])
	stateToObsMatrix = np.array([
						[0.5,0.5],
						[0.4,0.6],
						[0.7,0.3],
							])
	H = HMM(stateSet,obsSet,statePrior)
	H.stateTransMatrix = stateTransMatrix
	H.stateToObsMatrix = stateToObsMatrix
	H.lambda_=(stateTransMatrix,stateToObsMatrix,statePrior)
	obs = [0,1,0]
	# obsProb = H.forward(H.lambda_,obs,"forbackward")
	# print(obsProb)
	state = H.predict(H.lambda_,obs,"viterbe")
	print(list(state))
