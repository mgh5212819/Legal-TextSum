# coding=gbk
import re
import json
pattern = r'\?|!|。|；|！'

with open('input.json') as file:
	with open('test.json','w') as wf:
		lines = file.readlines()
		for index,line in enumerate(lines):
			text = re.split(pattern, line.strip())
			sentences = []
			for sentence in text:
				sentences.append({'sentence':sentence})
			wf.write(json.dumps({'id':index, 'text':sentences},ensure_ascii=False)+'\n')
	wf.close()
file.close()
			