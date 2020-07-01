import torch
import json
from utils import rouge, Vocab, OOVDict, Batch, format_tokens, format_rouge_scores, Dataset
from model import DEVICE, Seq2SeqOutput, Seq2Seq
import codecs
from utils import OOVDict
#change the input document and output document
input_path = '/input/input.json'
output_path = '/output/result.json'
from pytorch_pretrained_bert import BertModel,BertTokenizer
tokenizer = BertTokenizer.from_pretrained('pretrained_model/ms')


def decode_batch_output(decoded_tokens, vocab: Vocab, oov_dict: OOVDict):
  """Convert word indices to strings."""
  decoded_doc = []
  for i,word_idx in enumerate(decoded_tokens):
    if word_idx >= len(vocab):
      word = oov_dict.index2word.get((i,word_idx),'<UNK>')
    else:
      word = vocab[word_idx]
      decoded_doc.append(word)
      if word_idx == vocab.EOS:
        break
  return decoded_doc

def preprocess(text):
    sentences = ''
    for sentence in text:
        sentences += sentence['sentence']
    tokens = tokenizer.tokenize(sentences)
    return tokens
     
def generate_summary(text, vocab, model):
    tokens = preprocess(text)  
    lengths = [len(tokens)+1]  
    src_tensor = torch.zeros(lengths[0],1,dtype=torch.long)
    base_oov_idx = len(vocab)
    oov_dict = OOVDict(base_oov_idx)

    for index, word in enumerate(tokens):
        idx = vocab[word]
        if idx == vocab.UNK:
           idx = oov_dict.add_word(index,word)
        src_tensor[index] = idx
    src_tensor[lengths[0]-1,0] = vocab.EOS
    model.eval()   
    with torch.no_grad():
        src_tensor = src_tensor.to('cuda')
        hypotheses = model.beam_search(src_tensor,lengths,oov_dict.ext_vocab_size,min_out_len= 100)

    #print('hypotheses',hypotheses)
    decode_batch = decode_batch_output(hypotheses[0].tokens,vocab,oov_dict)
    res = format_tokens(decode_batch)
    return res

if __name__ == "__main__":
    train_status = torch.load('checkpoints/law.train.pt')
    filename = '%s.%02d.pt' % ('checkpoints/law', train_status['best_epoch_so_far'])
    print("Evaluating %s..." % filename)
    m = torch.load(filename)
    m.to('cuda')
    m.encoder.gru.flatten_parameters()
    m.decoder.gru.flatten_parameters()
    v = m.vocab    
    with open(output_path, 'w', encoding='utf8') as fw:
        with open(input_path, 'r', encoding="utf-8") as f:
            f = f.readlines()
            for line in f:
                #print('line',line)
                data = json.loads(line)
                id = data.get('id')
                text = data.get('text')  # "text": [{"sentence":"001"},{"sentence":"002"}]
                summary = generate_summary(text,v,m).replace(' ','' )
                print(summary)
                result = dict(
                    id=id,
                    summary=summary
                )
                fw.write(json.dumps(result, ensure_ascii=False) + '\n')