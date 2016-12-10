import sys
import os
import math
import string
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.externals import joblib
import numpy as np
from aligner import *
import pandas
import time
import skipthoughts

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
	# NewValue2 = (((OldValue - (OldMin-1)) * NewRange) / OldRange2) + NewMin
	# print NewValue , NewValue2
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
def get_alignment_similarity(FirstSentence,SecondSentence,Stopwords):
	FirstSentenceor = FirstSentence
	SecondSentenceor = SecondSentence
	table = string.maketrans("","")
	FirstSentence = FirstSentence.lower()
	SecondSentence = SecondSentence.lower()
	FirstSentence =FirstSentence.replace('/', ' ')
	SecondSentence =SecondSentence .replace('/', ' ')
	FirstSentence = FirstSentence.split(' ')
	SecondSentence = SecondSentence.split(' ')
	FirstSentenceFinal = []
	SecondSentenceFinal = []
	len1 = len(FirstSentence)
	len2 = len(SecondSentence)
	for i in range(0, len1):
		if(len(FirstSentence[i])>0 ):
			FirstSentence[i] = FirstSentence[i].translate(table, string.punctuation)
			if(FirstSentence[i].isalpha() and FirstSentence[i] not in Stopwords):
				FirstSentenceFinal.append(FirstSentence[i])

	for i in range(0, len2):
		if(len(SecondSentence[i])>0 ):
			SecondSentence[i] = SecondSentence[i].translate(table, string.punctuation)
			if(SecondSentence[i].isalpha() and SecondSentence[i] not in Stopwords):
				SecondSentenceFinal.append(SecondSentence[i])

	
	if(FirstSentenceFinal!=[] and SecondSentenceFinal != []):
		#print FirstSentenceFinal , SecondSentenceFinal
		alignments = align(FirstSentenceFinal, SecondSentenceFinal)
		similarity = (2*(float(len(alignments[0]))))/(len(FirstSentenceFinal)+len(SecondSentenceFinal))
	else:
		#print FirstSentenceor , SecondSentenceor
		alignments = align(FirstSentenceor, SecondSentenceor)
		similarity = (2*(float(len(alignments[0]))))/(len1+len2)
	
	# print FirstSentenceor , SecondSentenceor
	# print alignments
	# print similarity , len(alignments[0])
	# print  normalized(0,1,0,5,similarity)
	return normalized(0,1,0,5,similarity)

def get_processed_sentence(FirstSentence):
	table = string.maketrans("","")
	FirstSentence = FirstSentence.lower()
	FirstSentence =FirstSentence.replace('/', ' ')
	FirstSentence = FirstSentence.translate(table, string.punctuation)
	return FirstSentence

def get_sentence_similarity(FirstSentence,SecondSentence,Main_dictionary,Main_embeddings,dictionary,embeddings_dictionary):
	FirstSentenceor = FirstSentence
	SecondSentenceor = SecondSentence
	table = string.maketrans("","")
	FirstSentence = FirstSentence.lower()
	SecondSentence = SecondSentence.lower()
	FirstSentence =FirstSentence.replace('/', ' ')
	SecondSentence =SecondSentence .replace('/', ' ')
	FirstSentence = FirstSentence.split(' ')
	SecondSentence = SecondSentence.split(' ')
	len1 = len(FirstSentence)+1
	len2 = len(SecondSentence)+1
	FirstSentence_vector=[0] * 300
	SecondSentence_vector=[0] *  300
	num=0
	for i in range(0, len1-1):
		if(len(FirstSentence[i])>0):
			FirstSentence[i] = FirstSentence[i].translate(table, string.punctuation)

			vector = get_word_from_embeddings(FirstSentence[i],Main_dictionary,Main_embeddings,dictionary,embeddings_dictionary)
			if(vector != -1 and vector !=[]):
				num+=1
				FirstSentence_vector = [FirstSentence_vector[j] + vector[j] for j in range(300)]
	if(num > 1):
		FirstSentence_vector = [FirstSentence_vector[j]/num for j in range(300)]
	else:
		print FirstSentence
	num=0
	for i in range(0, len2-1):
		if(len(SecondSentence[i])>0):
			SecondSentence[i] = SecondSentence[i].translate(table, string.punctuation)
			# if(SecondSentence[i][-1] == '.' or SecondSentence[i][-1] == '?'):
			# 	SecondSentence[i]=SecondSentence[i][:-1]
			vector = get_word_from_embeddings(SecondSentence[i],Main_dictionary,Main_embeddings,dictionary,embeddings_dictionary)
			if(vector != -1 and vector !=[]):
				num+=1
				SecondSentence_vector = [SecondSentence_vector[j] + vector[j] for j in range(300)]
	if(num > 1):
		SecondSentence_vector = [SecondSentence_vector[j]/num for j in range(300)]
	else:
		print SecondSentence
	similarity = _cosine_similarity(FirstSentence_vector,SecondSentence_vector)
	# print similarity[0][0]
	return normalized(-1,1,0,5,similarity)

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
def get_Feature2(FirstSentences,SecondSentences,Stopwords):
	print "Feature 2"
	feature = []
	print len(FirstSentences)
	for i in range (len(FirstSentences)):
		#print FirstSentences[i] , SecondSentences[i]
		print i
		Sentencefeature = get_alignment_similarity(FirstSentences[i],SecondSentences[i],Stopwords)
		feature.append(Sentencefeature)
	return feature

def get_Feature3(FirstSentences,SecondSentences,model):
	print "Feature 3"
	feature = []
	PreprossedFirstSentences = []
	PreprossedSecondSentences = []
	for i in range (len(FirstSentences)):
		PreprossedFirstSentences . append(get_processed_sentence(FirstSentences[i]))
		PreprossedSecondSentences . append(get_processed_sentence(SecondSentences[i]))
	SentencesA = skipthoughts.encode(model, PreprossedFirstSentences, verbose=False, use_eos=True)
	SentencesB = skipthoughts.encode(model, PreprossedSecondSentences, verbose=False, use_eos=True)
	for i in range (len(SentencesA)):
		similarity = _cosine_similarity(SentencesA[i],SentencesB[i])
		Sentencefeature = normalized(-1,1,0,5,similarity)
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

def prepareModel(input,output):
	input = np.array(input)
	output = np.array(output)
	print input.shape
	print  output.shape
	#input = input.reshape(-1, 1)
	print input.shape
	#alphas = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
	model = RidgeCV()
	model.fit(input,output) 

	return model

def predict(model,input):
	input = np.array(input)
	#input = input.reshape(-1, 1)
	return model.predict(input)

# def mergeFeatures(f1,f2,f3):
# 	return np.column_stack((f1,f2,f3))

def mergeFeatures(f1,f2):
	return np.column_stack((f1,f2))
def eval(gold,predectvalues):
            pr = pearsonr(gold, predectvalues)[0]
            #print 'Test Pearson: ' + str(pr)
            return pr

if __name__ == "__main__":
	#create_dictionary("./paragram-phrase-XXL.txt","phrase-XXL_dictionary.txt")
	Main_dictionary = read_dictionary("../phrase-XXL_dictionary.txt")
	Main_embeddings = read_embeddings("../paragram-phrase-XXL.txt")

	
	Data = './../Data/'
	Stopwords = read_dictionary(Data + "Stopwords.txt")
	Output_path = './../Data/fancy_output/'
	WordEmbbedings = "/Users/aya/Documents/CMSC723/P3/paragram_300_sl999/paragram_300_sl999.txt"
	# BOW,unqiue_count = bag_of_words(Data +"2015train.input.txt")
	# create_training_embeddings(BOW,WordEmbbedings,Data + "2015train_embeddings.txt")
	# create_dictionary(Data +"2015train_embeddings.txt",Data +"2015train_dictionary.txt")
	# #
	# print BOW,unqiue_count
	# unk, unknown_count  = get_unknown_words(BOW,Main_dictionary,dictionary)
	# print unk , unknown_count

	ticstart = time.time()
   
	print "Training...."
	dictionary = read_dictionary(Data+"train_dictionary.txt")
	embeddings = read_embeddings(Data+"train_embeddings.txt")
	s1,s2 = read_data(Data+"train.input.txt")
	gs =  read_gs(Data+"train.gs.txt")
	#f1_train = get_Feature1(s1[:700],s2[:700],Main_dictionary,Main_embeddings,dictionary,embeddings)
	f2_train = get_Feature2(s1[:700],s2[:700],Stopwords)
	skModel = skipthoughts.load_model()
	f3_train = get_Feature3(s1[:700],s2[:700],skModel)
	train_features = mergeFeatures(f2_train,f3_train)
	

	model = prepareModel(train_features,gs[:700])
	filename = Data + 'f2andf3.sav'
	# # # # #model =joblib.load(filename) 

	joblib.dump(model, filename)
 
	# print('Coefficients: \n', model.coef_)
	# print model.intercept_

	# toc = time.time()
	# print('Processing time: %r'
 #       % (toc - ticstart))
	# os.system('say "Done training"')


	Datasets = ['answer-answer' , 'question-question','postediting','plagiarism','headlines']
	eval_DS = []
	#answer-answer , 'question-question','postediting','plagiarism','headlines',
	#Datasets = ['plagiarism']
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
		#f1_test = get_Feature1(s1,s2,Main_dictionary,Main_embeddings,dictionary,embeddings)
		f2_test = get_Feature2(s1,s2,Stopwords)
		f3_test = get_Feature3(s1,s2,skModel)
		test_features = mergeFeatures(f2_test,f3_test)
		yhat= predict(model,test_features)
		pr = eval(gs,yhat)
		prtotal+= pr
		print  dataset , " " , pr
		eval_DS.append(pr)
		# print_output(Output_path+"STS2016.out."+dataset+"_combined.txt",yhat)
		os.system('say "your dataset has finished"')
		toc = time.time()

	print prtotal
	print prtotal/5
	for i in range(len(Datasets)):
		print  Datasets[i] , " " , eval_DS[i]

	print('Processing time: %r'
	       % (toc - ticstart))
		

	os.system('say "your program has finished"')
