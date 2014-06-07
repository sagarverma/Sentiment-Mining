import csv
import nltk
def training_data_extract(filename,no):
	training_data = []
	count=0
	with open(filename,'Ur') as f:
		for rec in csv.reader(f,delimiter=','):
			count+=1
			if rec[0]=='0':
				training_data.append((rec[-1],'negative'))
			elif rec[0]=='2':
				training_data.append((rec[-1],'neutral'))
			elif rec[0]=='4':
				training_data.append((rec[-1],'positive'))
			if count==no:
				break
	return training_data

def test_data_extract(filename):
	test_data=[]
	with open(filename,'Ur') as f:
		for rec in csv.reader(f,delimiter=','):
			if rec[0]=='0':
				test_data.append((rec[-1],'negative'))
			elif rec[0]=='2':
				test_data.append((rec[-1],'neutral'))
			elif rec[0]=='4':
				test_data.append((rec[-1],'positive'))
	return test_data

def tweetizer(training_data):
	tweets=[]
	for (words,sentiment) in training_data:
		words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
		tweets.append((words_filtered, sentiment))
		
	return tweets

def get_words_in_tweets(tweets):
	all_words=[]
	for (words,sentiment) in tweets:
		all_words.extend(words)
	return all_words

def get_word_features(wordlist):
	worldlist=nltk.FreqDist(wordlist)
	word_features=worldlist.keys()
	return word_features


def extract_features(document):
	document_words = set(document)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features
	
#def main():
trainingfile='tdata.csv'
testfile='testdata.senti.csv'
no=2000
training_data=training_data_extract(trainingfile,no)
#print training_data[0:5]
tweets=tweetizer(training_data)
#print tweets[0:5]
word_features=get_word_features(get_words_in_tweets(tweets))
#print word_features[0:5]
training_set=nltk.classify.apply_features(extract_features,tweets)
#print training_set[0:5]
classifier=nltk.NaiveBayesClassifier.train(training_set)
while(1):
	tweet=raw_input("Enter your tweet.")
	print classifier.classify(extract_features(tweet.split()))
		
#if __name__=="__main__":
#	main()
