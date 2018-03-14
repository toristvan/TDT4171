import numpy as np
import math
import random

#examples is a matrix where each row is an example, and each column is an attribute
#attributes vary from 0-6 (1-7)
#parent_examples
#probability er gangene utfallet stemmer per verdi for attributen, og saa summert


class Tree:
	def __init__(self, att_type):
		self.att=att_type #what attribute does the tree split on
		self.subtrees={} #subtrees as dictionary, where the keys will be the different attribute values


test_matrix=np.full([28,8],-1,int)
examples=np.full([100,8],-1,int)
attributes=[i for i in range(0,np.size(examples,1)-1)]



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
				if(i==np.size(test_matrix,1)):
					j+=1
					i=0

def make_examplesmatrix():
	i=0
	j=0
	with open("training.txt") as test_file:
		for line in test_file:
			line.replace("\n","\t")
			attributes=line.split("\t")
			for att in attributes:
				examples[j][i]=att
				i+=1
				if(i==np.size(examples,1)):
					j+=1
					i=0

#for specified attribute type,
#for each value that attribute can take (here only 2)
#is the class positive? ->pk+=1
#att_type represents attribute type (0-6)
#att_value represents what value (1/2)
#defining 2 as positive, 1 as negative
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

#computing entropy of attribute, based on the probability of the two values
def B(q):
	if (q==0 or q==1):
		return 0
	return (-1*((q*math.log(q,2))+((1-q)*math.log(1-q,2))))


#valid when attribute can have value 1 or value 2
#computing ramining entropy
def Remainder(att_type, matrix):
	rem=0
	for i in range(1,3):
		pk=p_k(att_type,i,matrix)
		nk=n_k(att_type,i,matrix)
		ptot=p_total(matrix)
		ntot=n_total(matrix)
		q=pk/(pk+nk)
		rem+=((pk+nk)/(ptot+ntot))*B(q)
	return rem

#computing reduction in entropy for attribute type.
#The higher the reduction, the better the attribute is to split on
def Gain(att_type, matrix):
	ptot=p_total(matrix)
	ntot=n_total(matrix)
	q=ptot/(ptot+ntot)
	return (B(q)-Remainder(att_type,matrix))

#computing attribute with highest gain -> best attribute to split on
def importance_infogain(attributes_list, matrix):
	attributes=list(attributes_list)
	att_type_max=0
	gain_max=-1
	for att_type in attributes:
		gain=Gain(att_type,matrix)
		#print("Att:\t", att_type, "Gain:\t", gain) #to check the gain of each attribute
		if (gain>gain_max):
			att_type_max=att_type
			gain_max=gain
	return att_type_max

#giving each attribute random importance, and then chooses the best to split on
def importance_random(attributes_list):
	attributes=list(attributes_list)
	att_type_max=0
	gain_max=-1
	for att_type in attributes:
		gain=random.uniform(0,1)
		if (gain>gain_max):
			att_type_max=att_type
			gain_max=gain
	return att_type_max

#computing the most common classification in examples
def plurality_value(matrix):
	pos=p_total(matrix)
	neg=n_total(matrix)
	if (pos>neg):
		return 2
	elif (neg>pos):
		return 1
	else:
		return random.randint(1,2)

#checking if all the examples have the same classification
def same_classification(matrix):
	if (p_total(matrix)==np.size(matrix,0)):
		return 2
	elif (n_total(matrix)==np.size(matrix,0)):
		return 1
	else:
		return 0

#checking if matrix is valid
#using numpy, i specified the size beforehand, and filled them with -1's
#therefore i define valid as all entries having values >=0, and size being>0
def valid(matrix):
	if(np.size(matrix)==0):
		return False
	for i in matrix:
		for j in i:
			if(j<0):
				return False
	return True

#The decision tree learning-algorithm
def decision_tree_learning(examples, attributes_list, parent_examples, random_importance):
	print("list: ", attributes_list)
	attributes=list(attributes_list)
	if (not valid(examples)):
		print("examples not valid")
		return Tree(plurality_value(parent_examples))
	elif (same_classification(examples)>0):
		print("same classification")
		return Tree(same_classification(examples))
	elif (not attributes):
		print("no attributes")
		return Tree(plurality_value(examples))
	else:
		if (random_importance):
			A=importance_random(attributes)
			print("A_random:\t", A) #printing chosen attribute to split on
		else:
			A=importance_infogain(attributes, examples)
			print("A_infogain:\t", A) #printing chosen attribute to split on
		tree=Tree(A)
		attributes.remove(A)
		print("attributes: ", attributes)
		for vk in range(1,3):
			new_examples=[]
			for example in examples:
				if (example[A]==vk):
					new_examples.append(example)
			subtree=decision_tree_learning(new_examples,attributes,examples, random_importance)
			tree.subtrees[vk]=subtree
		return tree

#crating a copy of the test matrix, and replacing the classification
#with the one computed from the decision learning tree
def classify(tree, tests):
	classify_tests=np.copy(tests)
	for test in classify_tests:
		verif_tree=tree
		while(verif_tree.subtrees): #while the tree has subtrees
			verif_tree=verif_tree.subtrees[test[verif_tree.att]] #set the tree equal to its subtree corresponding to the value the test has for that attribute
			if (isinstance(verif_tree,int)):
				break
		test[-1]=verif_tree.att #tree.att will become the class value when there are noe more subtrees
	return classify_tests

#a direct way of testing how many of the classifications computed by the tree
#are correct. In this function, the classifcation isn't assigned to the examples.
#They are computed, by "moving" down the tree, then they are directly evaluated as
#correct or failed, without being assigned.
def decision_tree_test_direct(tree, tests):
	num_correct=0
	num_tests=0
	for test in tests:
		verif_tree=tree
		while(verif_tree.subtrees):
			verif_tree=verif_tree.subtrees[test[verif_tree.att]]
		if (verif_tree.att==test[-1]):
			num_correct+=1
		num_tests+=1
	print("Number of correct tests:\t", num_correct)
	print("Number of failed tests:\t", num_tests-num_correct)
	print("Total number of tests:\t", num_tests)

#using the classify function to check how many of the
#computed classifications that are correct
def decision_tree_test(tree, tests):
	classify_tests=classify(tree, tests)
	num_correct=0
	num_failed=0
	for i in range(0,np.size(tests,0)):
		if(classify_tests[i][-1]==tests[i][-1]):
			num_correct+=1
		else:
			num_failed+=1
	print("Out of ", num_correct+num_failed, " tests: ", num_correct, " were correct, and ", num_failed, "failed.")

def main():
	make_testmatrix() #making the test_matrix with data from "test.txt"
	make_examplesmatrix() #making the examples matrix with data from "training.txt"

	tree2=decision_tree_learning(examples, attributes,[],False) #building a tree with non-random importance of attributes.
	print("---Decision tree learning when importance is COMPUTED---")
	decision_tree_test(tree2, test_matrix)
	#decision_tree_test_direct(tree2, test_matrix)
	tree1=decision_tree_learning(examples, attributes,[],True)
	print("---Decision tree learning when importance is RANDOM---")
	decision_tree_test(tree1, test_matrix)
	#decision_tree_test_direct(tree1, test_matrix)



main()
