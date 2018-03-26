from rocketchat_API.rocketchat import RocketChat
import csv
import collections
import re
from pprint import pprint
from nltk.tag import StanfordNERTagger


channel_id='po5DghjugqfxwLahN' #Channel ID for gsoc2018
user=''#Insert username here
password=''#Insert password here
url='http://www.clips.uantwerpen.be:3001/' 

rocket = RocketChat(user, password, server_url=url)

#Storing all usernames from the channel in unames
channel_data=(rocket.channels_list().json())
for data in channel_data['channels']:
	if(data['_id']==channel_id):
		unames=(data['usernames'])
#print(unames)
#Pulling all messages from the start of session in 1st March 2018 15:30 IST
data=rocket.channels_history(channel_id, oldest='2018-03-01T10:02:30.156Z', count=2000).json()

count=0
datalist=[]
for msgs in data['messages']:
	if(msgs['msg'] not in unames): #Not considering channel join messages
		count+=1
		datalist.append(msgs['msg'])

print("Number of messages pulled: ",count)


'''----------------Preprocessing----------------'''

#Remove @mentions

hndl_regex = re.compile(r"@(\w+)")
def rem_hndl(text):
		text_split=text.split()
		for word in text_split:
			if(word==''):
				pass
			elif word[0]=='@':
				text_split.remove(word)
				
		text=' '.join(text_split)
		text=re.sub(hndl_regex,'',text)
		return text

url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")

def rem_url(text):
	text=re.sub(url_regex,'',text)
	return text

def rem_refer(text):
	text = str.replace(text,'[ ]','')
	#To fix a bug where message in between [ ]
	text = str.replace(text,']','')
	text = str.replace(text,'[','')
	text_split=text.split()

	for word in text_split:
		if(word==''):
			pass
		elif(len(word)==46 and word[0]=='(' and word[45]==')'): #46 is the lenght of refer text
			text_split.remove(word)
		text=' '.join(text_split)
	return text




'''----------------Stanford NER Tagger----------------'''
jar ='/Users/travis/Desktop/RChat NLP/stanford-ner-2018-02-27/stanford-ner.jar'
st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz',path_to_jar=jar)
def NER(text):
	text_split=text.split()
	NER_text = st.tag(text_split)
	for text in NER_text:
		if(text[1]!='O'):
			print(text)

#Calling all preprocessing functions on each line
for i in range(len(datalist)):
	datalist[i]=rem_hndl(datalist[i])
	datalist[i]=rem_url(datalist[i])
	datalist[i]=rem_refer(datalist[i])
	
	#NER(datalist[i])
#print(datalist)
resultFile = open("output.csv",'w')
for data in datalist:
	if(data!=None):
		resultFile.write(data + "\n")
    
resultFile.close()


