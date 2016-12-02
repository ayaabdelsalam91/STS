import sys
import math
import string
from sklearn.metrics.pairwise import cosine_similarity




def read_data(input):
	in_file = open(input, "r")
	FirstSentence = []
	SecondSentence = []
	for i,example in enumerate(in_file):
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

def get_word_from_glove(word,Main_dictionary,Main_glove,dictionary,glove):
	if(word in Main_dictionary):
		X_line = Main_glove[Main_dictionary[word]]
		X_vector = [float(num) for num in X_line[1:301]]
		if(X_vector ==[]):
			print "A" , X_line
		return X_vector
	elif(word in dictionary):
		X_line = glove[dictionary[word]]
		X_vector = [float(num) for num in X_line[1:301]]
		if(X_vector ==[]):
			print "B", X_line
			print dictionary[word]
		return X_vector
	else:
		return -1

def get_sentence_similarity(FirstSentence,SecondSentence,Main_dictionary,Main_glove,dictionary,glove_dictionary):
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
	for i in range(0, len1-1):
		if(len(FirstSentence[i])>0):
			FirstSentence[i] = FirstSentence[i].translate(table, string.punctuation)
			# if(FirstSentence[i][-1] == '.' or FirstSentence[i][-1] == '?'):
			# 	FirstSentence[i]=FirstSentence[i][:-1]
			vector = get_word_from_glove(FirstSentence[i],Main_dictionary,Main_glove,dictionary,glove_dictionary)
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
			vector = get_word_from_glove(SecondSentence[i],Main_dictionary,Main_glove,dictionary,glove_dictionary)
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

def output_similarity(out_path,FirstSentences,SecondSentences,Main_dictionary,Main_glove,dictionary,glove):
	out_file = open(out_path, "w")
	for i in range (len(FirstSentences)):
		#print FirstSentences[i] , SecondSentences[i]
		out_file.write(str(format(get_sentence_similarity(FirstSentences[i],SecondSentences[i],Main_dictionary,Main_glove,dictionary,glove))))
		out_file.write("\n")
	out_file.close()

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

def create_training_glove(BOW,in_path,out_path):
	in_file = open(in_path, "r")
	out_file = open(out_path, "w")
	for line in in_file:
		words = line.split()
		if(words[0] in BOW):
			#print BOW[words[0]] , words[0]
			out_file.write(line)
	in_file.close()
	out_file.close()

def read_glove (input):
	glove = []
	in_file = open(input, "r")
	X = in_file.readlines()
	for i, X_line in enumerate(X):
		old = X_line
		X_line = X_line.strip()
		X_line = X_line.split(' ')
		glove.append(X_line)
	print i
	in_file.close()
	return glove

if __name__ == "__main__":

	create_dictionary("paragram-phrase-XXL.txt","phrase-XXL_dictionary.txt")
	Main_dictionary = read_dictionary("phrase-XXL_dictionary.txt")
	Main_glove = read_glove("paragram-phrase-XXL.txt")

	BOW,unqiue_count = bag_of_words("datasets+scoring_script/test/STS2016.input.answer-answer.txt")
	create_training_glove(BOW,"paragram_300_sl999/paragram_300_sl999.txt","datasets+scoring_script/test/output/answer-answer_glove.txt")
	create_dictionary("datasets+scoring_script/test/output/answer-answer_glove.txt","datasets+scoring_script/test/output/answer-answer_glove_dictionary.txt")
	dictionary = read_dictionary("datasets+scoring_script/test/output/answer-answer_glove_dictionary.txt")
	# print BOW,unqiue_count
	# unk, unknown_count  = get_unknown_words(BOW,Main_dictionary,dictionary)
	# print unk , unknown_count
	glove = read_glove("datasets+scoring_script/test/output/answer-answer_glove.txt")
	s1,s2 = read_data("datasets+scoring_script/test/STS2016.input.answer-answer.txt")
	output_similarity("datasets+scoring_script/test/output/STS.output.answer-answer1.txt",s1,s2,Main_dictionary,Main_glove,dictionary,glove)

	BOW,unqiue_count = bag_of_words("datasets+scoring_script/test/STS2016.input.headlines.txt")
	create_training_glove(BOW,"paragram_300_sl999/paragram_300_sl999.txt","datasets+scoring_script/test/output/headlines_glove.txt")
	create_dictionary("datasets+scoring_script/test/output/headlines_glove.txt","datasets+scoring_script/test/output/headlines_glove_dictionary.txt")
	dictionary = read_dictionary("datasets+scoring_script/test/output/headlines_glove_dictionary.txt")
	# # print unqiue_count
	# # unk, unknown_count  = get_unknown_words(BOW,dictionary)
	# print unk , unknown_count
	glove = read_glove("datasets+scoring_script/test/output/headlines_glove.txt")
	s1,s2 = read_data("datasets+scoring_script/test/STS2016.input.headlines.txt")
	output_similarity("datasets+scoring_script/test/output/STS.output.headlines.txt",s1,s2,Main_dictionary,Main_glove,dictionary,glove)

	BOW,unqiue_count = bag_of_words("datasets+scoring_script/test/STS2016.input.plagiarism.txt")
	create_training_glove(BOW,"paragram_300_sl999/paragram_300_sl999.txt","datasets+scoring_script/test/output/plagiarism_glove.txt")
	create_dictionary("datasets+scoring_script/test/output/plagiarism_glove.txt","datasets+scoring_script/test/output/plagiarism_glove_dictionary.txt")
	dictionary = read_dictionary("datasets+scoring_script/test/output/plagiarism_glove_dictionary.txt")
	# print unqiue_count
	# unk, unknown_count  = get_unknown_words(BOW,dictionary)
	# print unk , unknown_count
	glove = read_glove("datasets+scoring_script/test/output/plagiarism_glove.txt")
	s1,s2 = read_data("datasets+scoring_script/test/STS2016.input.plagiarism.txt")
	output_similarity("datasets+scoring_script/test/output/STS.output.plagiarism.txt",s1,s2,Main_dictionary,Main_glove,dictionary,glove)

	BOW,unqiue_count = bag_of_words("datasets+scoring_script/test/STS2016.input.postediting.txt")
	create_training_glove(BOW,"paragram_300_sl999/paragram_300_sl999.txt","datasets+scoring_script/test/output/postediting_glove.txt")
	create_dictionary("datasets+scoring_script/test/output/postediting_glove.txt","datasets+scoring_script/test/output/postediting_glove_dictionary.txt")
	dictionary = read_dictionary("datasets+scoring_script/test/output/postediting_glove_dictionary.txt")
	# print unqiue_count
	# unk, unknown_count  = get_unknown_words(BOW,dictionary)
	# print unk , unknown_count
	glove = read_glove("datasets+scoring_script/test/output/postediting_glove.txt")
	s1,s2 = read_data("datasets+scoring_script/test/STS2016.input.postediting.txt")
	output_similarity("datasets+scoring_script/test/output/STS.output.postediting.txt",s1,s2,Main_dictionary,Main_glove,dictionary,glove)

	BOW,unqiue_count = bag_of_words("datasets+scoring_script/test/STS2016.input.question-question.txt")
	create_training_glove(BOW,"paragram_300_sl999/paragram_300_sl999.txt","datasets+scoring_script/test/output/question-question_glove.txt")
	create_dictionary("datasets+scoring_script/test/output/question-question_glove.txt","datasets+scoring_script/test/output/question-question_glove_dictionary.txt")
	dictionary = read_dictionary("datasets+scoring_script/test/output/question-question_glove_dictionary.txt")
	print unqiue_count
	unk, unknown_count  = get_unknown_words(BOW,Main_dictionary,dictionary)
	print unk , unknown_count
	glove = read_glove("datasets+scoring_script/test/output/question-question_glove.txt")
	s1,s2 = read_data("datasets+scoring_script/test/STS2016.input.question-question.txt")
	output_similarity("datasets+scoring_script/test/output/STS.output.question-question.txt",s1,s2,Main_dictionary,Main_glove,dictionary,glove)




