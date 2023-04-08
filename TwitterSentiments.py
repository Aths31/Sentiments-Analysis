from transformers import AutoTokenizer,AutoModelForSequenceClassification
from scipy.special import softmax

tweet = "@atharv60 wanna die 🥵 https://www.geeksforgeeks.org"

tweet_words = []

for word in tweet.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'
    elif word.startswith('https'):
        https = 'https'
    tweet_words.append(word)

tweet_process = " ".join(tweet_words)
print(tweet_process)

#Loading the model
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)

tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['negative','neutral','positive']

#tweets sentiments analysis

encoded_tweet = tokenizer(tweet_process,return_tensors='pt')

output = model(**encoded_tweet)

scores = output[0][0].detach().numpy()
scores = softmax(scores)

for i in range(len(scores)):
    l = labels[i]
    s = scores[i]
    print(l,":",s)

if(scores[0]>scores[1] and scores[0]>scores[2]):
    print("The entered tweet is ",labels[0])
elif(scores[1]>scores[0] and scores[1]>scores[2]):
    print("The entered tweet is ",labels[1])
elif(scores[2]>scores[0] and scores[2]>scores[1]):
    print("The entered tweet is ",labels[2])