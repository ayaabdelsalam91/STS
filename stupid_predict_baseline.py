import sys
import os
import math
import string
from scipy.stats import pearsonr
from unicodedata import category

import time


def read_data(input):
	in_file = open(input, "r")
	FirstSentence = []
	SecondSentence = []

	for i,example in enumerate(in_file):
		# print example
		s1, s2 = example.strip().split('\t')
		FirstSentence.append(s1)
		SecondSentence.append(s2)
	return FirstSentence,SecondSentence
	in_file.close()

def _cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def normalized(OldMin,OldMax,NewMin,NewMax,OldValue):
	OldRange = (OldMax - OldMin)  
	# OldRange2 = (OldMax - (OldMin -1))
	# print OldRange2  
	NewRange = (NewMax - NewMin)  
	NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
	return NewValue

def get_word_from_embeddings(word,Main_dictionary,Main_embeddings,dictionary,embeddings):
	# print len(dictionary)
	# print len(embeddings)
	if(word in Main_dictionary):
		X_line = Main_embeddings[Main_dictionary[word]]
		X_vector = [float(num) for num in X_line[1:301]]
		if(X_vector ==[]):
			print "A" , X_line
		
		return X_vector
	elif(word in dictionary):
		X_line = embeddings[dictionary[word]]
		X_vector = [float(num) for num in X_line[1:301]]
		if(X_vector ==[]):
			print "B", X_line
			print dictionary[word]
		
		return X_vector
	else:
		return -1


def get_sentence_similarity(FirstSentence,SecondSentence,Main_dictionary,Main_embeddings,dictionary,embeddings_dictionary):
	FirstSentenceor = FirstSentence
	SecondSentenceor = SecondSentence
	table = string.maketrans("","")
	FirstSentence = FirstSentence.lower()
	SecondSentence = SecondSentence.lower()
	FirstSentence = FirstSentence.split(' ')
	SecondSentence = SecondSentence.split(' ')
	len1 = len(FirstSentence)+1
	len2 = len(SecondSentence)+1
	FirstSentence_vector=[0] * 300
	SecondSentence_vector=[0] *  300
	num=0
	flag = False
	for i in range(0, len1-1):
		if(len(FirstSentence[i])>0):
			FirstSentence[i] = FirstSentence[i].translate(table, string.punctuation)

			vector = get_word_from_embeddings(FirstSentence[i],Main_dictionary,Main_embeddings,dictionary,embeddings_dictionary)
			if(vector != -1 and vector !=[]):
				num+=1
				FirstSentence_vector = [FirstSentence_vector[j] + vector[j] for j in range(300)]
	if(num > 1):
		FirstSentence_vector = [FirstSentence_vector[j]/num for j in range(300)]
	elif(num == 0 ):
		flag = True
	num=0
	for i in range(0, len2-1):
		if(len(SecondSentence[i])>0):
			SecondSentence[i] = SecondSentence[i].translate(table, string.punctuation)
			vector = get_word_from_embeddings(SecondSentence[i],Main_dictionary,Main_embeddings,dictionary,embeddings_dictionary)
			if(vector != -1 and vector !=[]):
				num+=1
				SecondSentence_vector = [SecondSentence_vector[j] + vector[j] for j in range(300)]
	if(num > 1):
		SecondSentence_vector = [SecondSentence_vector[j]/num for j in range(300)]
	elif(num == 0 ):
		flag = True
	if(not flag):
		similarity = _cosine_similarity(FirstSentence_vector,SecondSentence_vector)
	else:
		similarity =-1
	return normalized(0.7,1,0,5,similarity)

def output_similarity(out_path,FirstSentences,SecondSentences,Main_dictionary,Main_embeddings,dictionary,embeddings):
	out_file = open(out_path, "w")
	for i in range (len(FirstSentences)):
		#print FirstSentences[i] , SecondSentences[i]
		out_file.write(str(format(get_sentence_similarity(FirstSentences[i],SecondSentences[i],Main_dictionary,Main_embeddings,dictionary,embeddings))))
		out_file.write("\n")
	out_file.close()

def print_output(out_path,outputs):
	out_file = open(out_path, "w")
	for output in outputs:
		#print FirstSentences[i] , SecondSentences[i]
		out_file.write(str(format(output)))
		out_file.write("\n")
	out_file.close()

def get_Feature1(FirstSentences,SecondSentences,Main_dictionary,Main_embeddings,dictionary,embeddings):
	# print "Feature 1"
	feature = []
	# print len(FirstSentences)
	for i in range (len(FirstSentences)):
		#print FirstSentences[i] , SecondSentences[i]
		# print i
		Sentencefeature = get_sentence_similarity(FirstSentences[i],SecondSentences[i],Main_dictionary,Main_embeddings,dictionary,embeddings)
		feature.append(Sentencefeature)
	return feature

def bag_of_words(path):
	file = open(path, "r")
	table = string.maketrans("","")
	BOW = dict()
	unqiue_count=0
	for line in file:
		words = line.split()
		for word in words:
			#print word
			word = word.translate(table, string.punctuation)
			# if(word[-1]=='.' or word[-1]=='?'):
			# 	word=word[:-1]
			if word in BOW:
				BOW[word.lower()] += 1
			else:
				BOW[word.lower()] = 1
				unqiue_count+=1
	file.close()
	return BOW,unqiue_count

def create_dictionary(in_path,out_path):
	in_file = open(in_path, "r")
	out_file = open(out_path, "w")
	for line in in_file:
		words = line.split()
		out_file.write(words[0])
		out_file.write("\n")
	in_file.close()
	out_file.close()

def read_dictionary(in_path):
	# print in_path
	_dictionary = dict()
	in_file = open(in_path, "r")
	for i,line in  enumerate(in_file):
		word = line.split()
		_dictionary[word[0]] = i
	in_file.close()
	return _dictionary

def get_unknown_words(BOW,Main_dictionary,dictionary):	
	unk=[]
	unknown_count=0
	for word in BOW:
		if((word not in dictionary) and (word not in Main_dictionary)):
			unk.append(word)
			unknown_count+=1
	return unk, unknown_count

def create_training_embeddings(BOW,in_path,out_path):
	in_file = open(in_path, "r")
	out_file = open(out_path, "w")
	for line in in_file:
		words = line.split()
		if(words[0] in BOW):
			#print BOW[words[0]] , words[0]
			out_file.write(line)
	in_file.close()
	out_file.close()

def read_embeddings (input):
	# print input
	embeddings = []
	in_file = open(input, "r")
	X = in_file.readlines()
	for i, X_line in enumerate(X):
		old = X_line
		X_line = X_line.strip()
		X_line = X_line.split(' ')
		embeddings.append(X_line)
	#print i
	in_file.close()
	return embeddings

def read_gs (input):
	gs = []
	in_file = open(input, "r")
	X = in_file.readlines()
	for i, X_line in enumerate(X):
		X_line = X_line.strip()
		gs.append(X_line)
	#print i
	in_file.close()
	gs = [float(s) for s in gs]
	return gs

def eval(gold,predectvalues):
            pr = pearsonr(gold, predectvalues)[0]
            #print 'Test Pearson: ' + str(pr)
            return pr

if __name__ == "__main__":
	#create_dictionary("./paragram-phrase-XXL.txt","phrase-XXL_dictionary.txt")
	Main_dictionary = read_dictionary("./phrase-XXL_dictionary.txt")
	Main_embeddings = read_embeddings("./paragram-phrase-XXL.txt")

	
	Data = './Data/'
	Output_path = './Data/Basic_output/'
	WordEmbbedings = "/Users/aya/Documents/CMSC723/P3/paragram_300_sl999/paragram_300_sl999.txt"
	# BOW,unqiue_count = bag_of_words(Data +"2015train.input.txt")
	# create_training_embeddings(BOW,WordEmbbedings,Data + "2015train_embeddings.txt")
	# create_dictionary(Data +"2015train_embeddings.txt",Data +"2015train_dictionary.txt")
	# #
	# print BOW,unqiue_count
	# unk, unknown_count  = get_unknown_words(BOW,Main_dictionary,dictionary)
	# print unk , unknown_count

	Datasets = ['answer-answer' , 'question-question','postediting','plagiarism','headlines']
	eval_DS = []
	#answer-answer , 'question-question','postediting','plagiarism','headlines',
	#Datasets = ['headlines']
	print "Testing...."
	prtotal = 0 
	for dataset in Datasets:
		print "Testing "  + dataset
		tic = time.time()
		# BOW,unqiue_count = bag_of_words(Data +"STS2016.input."+dataset+".txt")
		# create_training_embeddings(BOW,WordEmbbedings,Data+dataset+ "_embbedings.txt")
		# create_dictionary(Data+dataset+ "_embbedings.txt",Data+dataset+ "_dictionary.txt")
		dictionary = read_dictionary(Data+dataset+"_dictionary.txt")
		embeddings = read_embeddings(Data+dataset+ "_embbedings.txt")
		s1,s2 = read_data(Data +"STS2016.input."+dataset+".txt")
		gs =  read_gs(Data +"STS2016.gs."+dataset+".txt")
		f1_test = get_Feature1(s1,s2,Main_dictionary,Main_embeddings,dictionary,embeddings)
		pr = eval(gs,f1_test)
		prtotal+= pr
		print  dataset , " " , pr
		eval_DS.append(pr)
		print_output(Output_path+"STS2016.out."+dataset+".txt",f1_test)
		os.system('say "your dataset has finished"')
		toc = time.time()

	print prtotal
	print prtotal/5
	for i in range(len(Datasets)):
		print  Datasets[i] , " " , eval_DS[i]

	print('Processing time: %r'
	       % (toc - ticstart))
		

	os.system('say "your program has finished"')
