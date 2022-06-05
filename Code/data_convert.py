import pandas
import json
import hashlib
import nltk
import unidecode
from fuzzywuzzy import fuzz

nltk.download('punkt')

df = pandas.read_csv('./bidaf-keras-bert/final-dataset/test.csv', sep = '\t')
jsonFilePath = './bidaf-keras-bert/final-dataset/test-v1.1.json'

# dfstart = pandas.read_csv('./data/our-dataset-span.csv', header = None, sep = '\t')
ix = 0;
removed = 0

def generateHash(df):
	str = df['Context'] + df['Answer'] + df['Question']
	return hashlib.md5(str.encode()).hexdigest()

def findStartSpan(context, answer, question):
	sentencelist =  nltk.tokenize.sent_tokenize(context)
	s = []
	for idx, val in enumerate(sentencelist):
		if (val.find(answer) == -1):
			pass
		else:
			s.append(idx)
	if len(s) == 0:
		# print(context)
		# print(answer)
		return -1
	tokenScore = []
	for indexvalue in s:
		tokenScore.append(fuzz.token_set_ratio(sentencelist[indexvalue], question))

	occurrence = tokenScore.index(max(tokenScore)) + 1
	inilist = [i for i in range(0, len(context)) if context[i:].startswith(answer)]
	return inilist[occurrence-1]

def useString(mystr):
	# mystr = str(mystr)
	# print(mystr)
	mystr = mystr.strip(' ,.*"')
	mystr = unidecode.unidecode(mystr)
	mystr = mystr.replace('-', ' ')
	mystr = ' '.join(mystr.split())
	# mystr.replace(/[^\x00-\x7F]/g, "")
	return mystr

root_data = {
	"version": "1.1",
	"data":[]
}

passageCount = 0;
passageArray = [];
currPassage = df.iloc[0][0];
currCount = 1;

for i in range(0, df.shape[0] - 1):
	if (df.iloc[i+1, 0] == currPassage):
		currCount = currCount + 1;
	else:
		passageArray.append(currCount);
		currPassage = df.iloc[i+1, 0];
		currCount = 1;

passageArray.append(currCount);

_sum = 0
for i in range (0, len(passageArray)):
	_sum = _sum + passageArray[i]
	currObject = {
		"title": "",
		"paragraphs":[]
	}
	currContext = {
		"context": useString(df.iloc[ _sum - 1, 0]),
		"qas":[]
	}


	for j in range(_sum - passageArray[i], _sum):
		if findStartSpan(useString(df.iloc[ _sum - 1, 0]), useString(df.iloc[j, 1]), useString(df.iloc[j, 2])) == -1:
			removed = removed + 1
			ix = ix + 1
			continue
		currQues = {
			"question": useString(df.iloc[j, 2]),
			"answers":[
				{
					"text": useString(df.iloc[j, 1]),
					"answer_start": findStartSpan(useString(df.iloc[ _sum - 1, 0]), useString(df.iloc[j, 1]), useString(df.iloc[j, 2]))
					# "answer_start": int(dfstart.iloc[ix, 0])
				}
			],
			"id": generateHash(df.iloc[j, :])
		}
		ix = ix + 1
		currContext["qas"].append(currQues)

	currObject["paragraphs"].append(currContext)

	root_data["data"].append(currObject)

with open(jsonFilePath, 'w') as jsonFile:
	jsonFile.write(json.dumps(root_data, indent = 4))

print("Done!")
print("Removed: "+str(removed))