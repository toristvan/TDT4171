import numpy as np

#1) Tranisition model, evidence matrix and initial probability of rain.
#	The evidence matrix is called with either O_t or O_f (see declaration in main)
# 	based on evidence presented.
#2) transpose of the transition model T=P(X_t|X_t-1)
#3) Probability of rain on given day, based on rain previous day.
#4) Probability of rain given evidence observed.
#5) Normalizes f_next

#forward2 is a function that takes in a vector with evidences,
# and computes probability of rain the last day.
# a similar loop is implemented in main, so it should not be necessary to run.
def forward2(T,O_t,O_f,n, prior):
	P=prior
	T_trans=np.transpose(T)
	for i in range(0,n.size):
		P=np.dot(T_trans,P)
		if(n[i]==1):
			f_next=np.dot(O_t,P)
		elif(n[i]==0):
			f_next=np.dot(O_f,P)
		alpha=1/(f_next[0]+f_next[1])
		f_next=alpha*f_next
		print('P(X_',i+1,')')
                print(f_next)
		P=f_next
	print('Probability of rain after evidence no: ',i+1)
	return P

def forward(T,ev,prior): 						#1
	T_trans=np.transpose(T)						#2
	P=np.dot(T_trans,prior)						#3
	f_next=np.dot(ev,P)							#4
	alpha=1/(f_next[0]+f_next[1])				#5
	f_next=alpha*f_next
	return f_next

#6-8 Creating forward vector, vector of smoothed estimated steps and vector of backward messages.
#	forward vector is +1 larger in size to store "prior"
#9-10 Initializing backward message (that has yet to be observed, and is therefore 1's),
#	and sets first forward message=initial probability of rain.
#11	 Computes all forward messages
#12 Computes smoothes estimated steps
# 13 Computes backward messages. Probability of bringing umbrella the next day if it rained previous day,
# and if it didnt rain previous day. b=[U_2|r_2,U_2|!R_2]
#first iteration, e_t+1:t is an empty sequence=1 (it hasnt happened yet)
#next iterations will represent the probability of bringing the umbrella every day
#or not bringing the umbrella every day, from day t-iteration until day t, based on wether
#it rained on not on day t-iteration-1, and we will use the previously computed values for b
#to compute this.
#14-16 prints fv, b and sv

def forwardbackward(T,O_t,O_f,n, prior):
	fv=np.empty([n.size+1,2],dtype=float,order='C')		#6
	sv=np.empty([n.size,2],dtype=float,order='C')		#7
	bv=np.empty([n.size,2])								#8
	b=np.array([1.0,1.0])								#9
	fv[0]=prior											#10
	for i in range(1,n.size+1):							#11
		temp=fv[i-1]
		if (n[i-1]==1):
			fv[i]=(forward(T,O_t,temp))
		if (n[i-1]==0):
			fv[i]=(forward(T,O_f,temp))
	for j in range(n.size,0,-1):						#12
		f_x_b=fv[j]*b
		f_x_b=f_x_b/(f_x_b[0]+f_x_b[1])
		sv[j-1]=f_x_b
		if (n[j-1]==1):									#13
			b=np.dot(np.dot(T,O_t),b)
		elif(n[j-1]==0):								#13
			b=np.dot(np.dot(T,O_f),b)
		bv[n.size-j]=b #j-1
	print("fv: ")										#14
	print(fv)
	print("b: ")										#15
	print(bv)
	print("sv: ")										#16
	return sv

def main():
	T=np.array([[0.7,0.3],[0.3,0.7]])
	O_t=np.array([[0.9,0],[0,0.2]])
	O_f=np.array([[0.1,0],[0,0.8]])
	prior=np.array([0.5,0.5])
	n=np.array([1,1,0,1,1])
	m=np.array([1,1])
	#print(forward2(T,O_t,O_f,n,prior))
	print(forwardbackward(T,O_t,O_f,n,prior))

"""
	#Prints all forward messages for given evidence vector n.
	fv=np.empty([n.size+1,2])
	fv[0]=prior
	for i in range(1,n.size+1):
		temp=fv[i-1]
		if (n[i-1]==1):
			fv[i]=(forward(T,O_t,temp))
		if (n[i-1]==0):
			fv[i]=(forward(T,O_f,temp))
	print(fv)
"""



main()
