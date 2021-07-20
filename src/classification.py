from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences

import torch

if torch.cuda.is_available():
  device = torch.device("cuda")
  print('There are %d GPU(s) available.' % torch.cuda.device_count())
  print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
  device = torch.device("cpu")
  print('No GPU available, using the CPU instead.')

def evaluate(text, e_model, e_tokenizer):
  e_model.to(device)
  e_model.eval()

  m = {
      0: 'A type',
      1: 'B type',
      2: 'C type',
      3: 'D type'
  }
  MAX_LEN = 256

  sentence = text
  sentences = ["[CLS] " + sentence + " [SEP]"]

  tokenized_texts = [e_tokenizer.tokenize(sent) for sent in sentences]

  input_ids = [e_tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
  input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

  attention_masks = []
  for seq in input_ids:
      seq_mask = [float(i>0) for i in seq]
      attention_masks.append(seq_mask)

  test_inputs = torch.tensor(input_ids).to(device)
  test_masks = torch.tensor(attention_masks).to(device)

  outputs = e_model(test_inputs, token_type_ids=None, attention_mask=test_masks)
  
  # 로스 구함
  logits = outputs[0]

  # CPU로 데이터 이동
  logits = logits.detach().cpu().numpy()
  print('this text predicted :', m[logits.argmax(1)[0]])
  return logits.argmax(1)[0]

if __name__ == "__main__":
  m_model = BertForSequenceClassification.from_pretrained("/content/drive/MyDrive/SAI/p/lunab_model_21_07_14_1")
  m_tokenizer = BertTokenizer.from_pretrained("/content/drive/MyDrive/SAI/p/lunab_tokenizer_21_07_14_1")

  evaluate("i feel the wind while looking at the night sky in summer.", m_model, m_tokenizer)
  evaluate("it's like having a strong spice in your mouth.", m_model, m_tokenizer,)