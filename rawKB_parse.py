import argparse
from tqdm import tqdm
import argparse
import pandas as pd
import re
import os
pattern = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+")

parser = argparse.ArgumentParser()
parser.add_argument('-data_path', action='store', dest='data_path', help='data path to file', default='./data')
parser.add_argument('-fileName', action='store', dest='fileName', help='file name', default='raw_kb.xlsx')
parser.add_argument('-type', action='store', dest='type', help='q for question, a for answer', default='a')
params = parser.parse_args()


kb = pd.read_excel(os.path.join(params.data_path, params.fileName), header=None)
kb_parsed = pd.DataFrame(columns=['category', 'question', 'answer'])
for index in tqdm(kb.index, desc="{Read questions from raw knowledge base data}"):
	answer = pattern.sub("", kb.loc[index, 3])
	question = pattern.sub("", kb.loc[index, 2])
	category = pattern.sub("", kb.loc[index, 0])
	if not answer or not question or not category: continue
	kb_parsed = kb_parsed.append({'category': category, 'question':question, 'answer':answer}, ignore_index=True)

	cur = kb.loc[index, :].dropna()
	for i in range(12, cur.index[-1]):
		sim_question = pattern.sub("" ,cur.loc[i])
		if not sim_question: continue
		kb_parsed = kb_parsed.append({'category': category, 'question':sim_question, 'answer':answer}, ignore_index=True)

print ("output kb_parsed.csv ...")
kb_parsed.to_csv(os.path.join(params.data_path, "kb_parsed.csv"), encoding='utf_8_sig', index=False)
