counter=0
def print_tree(tree, counter):
	cnt=counter+1
	print(tree.att)
	if(tree.subtrees[1].subtrees):
		print("att_val: 1")
		print("th: ", cnt)
		print_tree(tree.subtrees[1], cnt)
	else:
		print(tree.subtrees[1].att)
	if(tree.subtrees[2].subtrees):
		print("att_val: 2")
		print("th: ", cnt)
		print_tree(tree.subtrees[2], cnt)
	else:
		print(tree.subtrees[2].att)
