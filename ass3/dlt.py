import numpy as np
import math

#examples is a matrix where each row is an example, and each column is an attribute
#attributes vary from 0-6 (1-7)
#parent_examples
#probability er gangene utfallet stemmer per verdi for attributen, og saa summert


examplesss=np.empty([28,8],int)
parent_examples=np.empty([28,8],int)
test_matrix=np.empty([28,8],int)
training_matrix=np.empty([100,8],int)

def make_testmatrix():
	i=0
	j=0
	with open("test.txt") as test_file:
		for line in test_file:
			line.replace("\n","\t")
			attributes=line.split("\t")
			for att in attributes:
				test_matrix[j][i]=att
				i+=1
				if(i==8):
					j+=1
					i=0

def make_trainingmatrix():
	i=0
	j=0
	with open("training.txt") as test_file:
		for line in test_file:
			line.replace("\n","\t")
			attributes=line.split("\t")
			for att in attributes:
				training_matrix[j][i]=att
				i+=1
				if(i==8):
					j+=1
					i=0

#for bestemt type attribute, for bestemt verdi av attribute (her kun 2)
#er utfallet positivt?=pk+=1
#att_type sier hvilken attribute (0-6)
#att_value sier hvilket subset (1/2)
#definerer 2 som positivt, 1 som negativt
def p_k(att_type, att_value, matrix):
	pk=0.0
	for i in range(0,np.size(matrix,0)):
		if(matrix[i][att_type]==att_value):
			if(matrix[i][np.size(matrix,1)-1]==2):
				pk+=1.0
	return pk

def n_k(att_type, att_value, matrix):
	nk=0.0
	for i in range(0,np.size(matrix,0)):
		if (matrix[i][att_type]==att_value):
			if(matrix[i][np.size(matrix,1)-1]==1):
				nk+=1.0
	return nk

def p_total(matrix):
	ptotal=0.0
	for i in range(0,np.size(matrix,0)):
		if (matrix[i][np.size(matrix,1)-1]==2):
			ptotal+=1.0
	return ptotal

def n_total(matrix):
	ntotal=0.0
	for i in range(0,np.size(matrix,0)):
		if (matrix[i][np.size(matrix,1)-1]==1):
			ntotal+=1.0
	return ntotal
#q= p/p+n
def B(q):
	return (-1*((q*math.log(q,2))+((1-q)*math.log(1-q,2))))


#valid when attribute can have value 1 or value 2
def Remainder(att_type, matrix):
	rem=0
	for i in range(1,3):
		pk=p_k(att_type,i,matrix)
		nk=n_k(att_type,i,matrix)
		ptot=p_total(matrix)
		ntot=n_total(matrix)
		q=pk/(pk+nk)
		if(pk!=0 and nk!=0):
			rem+=((pk+nk)/(ptot+ntot))*B(q)
	return rem

#dobbeltsjekk matten
def Gain(att_type, matrix):
	ptot=p_total(matrix)
	ntot=n_total(matrix)
	q=ptot/(ptot+ntot)
	return (B(q)-Remainder(att_type,matrix))

#def importance_infogain(a, examples):



def main():
	make_testmatrix()
	make_trainingmatrix()
	#print("--test_matrix--")
	#print(test_matrix)
	#print("--training_matrix--")
	#print(training_matrix)
	pk=p_k(0,2,test_matrix)
	nk=n_k(0,2,test_matrix)
	ptot=p_total(test_matrix)
	ntot=n_total(test_matrix)
	print("Remainder:", Remainder(0,test_matrix))
	print(Gain(0,test_matrix))
main()
