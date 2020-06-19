#-*- encoding:utf-8 -*-
from __future__ import print_function
import json
import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import codecs
from textrank4zh import TextRank4Sentence


    

input_path = "../input.json"
output_path = "./result.json"


def get_summary(text):
    all_text =""
    for i, _ in enumerate(text):
        sent_text = text[i]["sentence"]
        all_text = all_text + sent_text
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=all_text, lower=True, source = 'all_filters')
    result = ""
    for item in tr4s.get_key_sentences(num=3):
        result = result + item.sentence
   

    return result    
    
    


if __name__ == "__main__":
    with open(output_path, 'w', encoding='utf8') as fw:
        with open(input_path, 'r', encoding="utf8") as f:
            for line in f:
                
                data = json.loads(line)
                id = data.get('id')
                text = data.get('text')  # "text": [{"sentence":"001"},{"sentence":"002"}]
                summary = get_summary(text)  # your model predict
                result = dict(
                    id=id,
                    summary=summary
                )

                fw.write(json.dumps(result, ensure_ascii=False) + '\n')