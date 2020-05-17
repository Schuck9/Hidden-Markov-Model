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
			stateProb = np.zeros(self.stateNum)
			# for t in range(len(obs)):
			# 	print("t == ",t)
			# 	for i,s in enumerate(self.stateSet):
			# 		if t == 0:
			# 			stateProb[i] = self.statePrior[s]*self.stateToObsMatrix[s][obs[t]]
			# 			print("state == {}, stateProb == {}".format(s,stateProb[i]))
			# 		else:
			# 			sumProb = 0
			# 			for s_ in stateSet:
			# 				sumProb += stateProb[s_]*stateTransMatrix[s_][s]
			# 			print(sumProb,stateToObsMatrix[s][obs[t]])
			# 			stateProb[i] = sumProb*self.stateToObsMatrix[s][obs[t]]
			# 			print("state == {}, stateProb == {}".format(s,stateProb[s]))
			
			#matrix caculation
			stateProb = np.ones(self.stateNum)
			for t in range(len(obs)):
				if t == 0:
					stateProb *= self.statePrior*stateToObsMatrix[:][:,obs[t]]
					# for s in self.stateSet:
					# 	print("state == {}, stateProb == {}".format(s,stateProb[s]))
				else:
					stateProb = np.dot(stateProb,stateTransMatrix)
					stateProb *= stateToObsMatrix[:][:,obs[t]]
					# for s in self.stateSet:
					# 	print("state == {}, stateProb == {}".format(s,stateProb[s]))

			obsProb = sum(stateProb)
			return obsProb 

		

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
	obsProb = H.forward(H.lambda_,obs)
	print(obsProb)