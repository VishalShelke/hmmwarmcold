
from __future__ import division
import numpy as np
import random
from sklearn import hmm

states = ["Warm", "Cold"]
n_components = len(states)

observations = ["jacket", "Sweater", "Tshirt"]
n_observations = len(observations)

startprob = np.array([0.7, 0.3])

transmat = np.array([
  [0.7, 0.3],
  [0.2, 0.8]
])


emissionprob = np.array([
  [0.2, 0.3, 0.5],
  [0.6, 0.3, 0.1]
])
s="\nvalue of ncomponents"+str(n_components)+"\n value of startprob"+str(startprob)+"\nvalue of transmat"+str(transmat)+"\nvalue of emissionprob"+str(emissionprob)
print s

model = hmm.MultinomialHMM(n_components=n_components)
model._set_startprob(startprob)
model._set_transmat(transmat)
model._set_emissionprob(emissionprob)
#k,obj=model.sample(3)
k,obj=model.sample(3,random_state=1)# seed 1 and samples 3
print"\nsample state is"
print(k)
# predict a sequence of hidden states based on visible states
bob_wears = [0, 2, 2]
logprob, alice_predicts = model.decode(bob_wears, algorithm="viterbi")
print "Bob wears:", ", ".join(map(lambda x: observations[x], bob_wears))# prediction for given sample
print "Alice predicts:", ", ".join(map(lambda x: states[x], alice_predicts))
#np.random.seed( seed=1 )
#bob_wears1 = np.random.randint(3, size=3)
#logprob, alice_predicts1 = model.decode(bob_wears1, algorithm="viterbi")
logprob, alice_predicts1 = model.decode(k, algorithm="viterbi")
print "\nBob wears:", ", ".join(map(lambda x: observations[x], k))# prediction for  random samples
print "Alice predicts:", ", ".join(map(lambda x: states[x], alice_predicts1))
