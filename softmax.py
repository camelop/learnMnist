import math as mt
p = [2.,4.,6.,8.,10.]
ep = []
for i in p:
	ep.append(mt.exp(i))
sum = 0
for i in ep:
	sum += i
gl = []
for i in ep:
	gl.append(i/sum)
for i in gl:
	print(i)