def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

def findConsecutiveSame(heldoutB, trainB, heldoutP=None, trainP=None):
	with open(heldoutB, 'r') as hb:
		heldoutBList = hb.readlines()
	with open(heldoutP, 'r') as hp:
		heldoutPList = hp.readlines()
	with open(trainB, 'r') as tb:
		trainBList = tb.readlines()
	with open(trainP, 'r') as tp:
		trainPList = tp.readlines()
	lineNoB = list()
	lineNoP = list()
	for idx, hLine in enumerate(heldoutBList):
		if hLine in trainBList:
			lineNoB.append(idx)
	for idx, hLine in enumerate(heldoutPList):
		if hLine in trainPList:
			lineNoP.append(idx)

	intersec = intersection(lineNoB, lineNoP)

	with open ("duplicate_indices_test.txt", 'w') as di:
		for l in intersec:
			di.write(str(l) + "\n")

findConsecutiveSame("Data/Analysis/test-buggy.txt", "Data/Analysis/train-buggy.txt", "Data/Analysis/test-fixed-filtered.txt", "Data/Analysis/train-fixed-filtered.txt")