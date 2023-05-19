from transformers import BertTokenizer
from transformers.models.bart.modeling_bart import BartForConditionalGeneration,BartConfig
from scripts.models import BartForConditionalGenerationWithCopyMech

tokenizer = BertTokenizer.from_pretrained("./tokenizer/")
model_1 = BartForConditionalGeneration.from_pretrained("./pretrain_model/ngram_mlm_seq2seq/")
model_2 = BartForConditionalGeneration.from_pretrained("./pretrain_model/ngram_mlm_seq2seq/")

from scripts.ensemble import Ensemble
G = Ensemble(model_1.config,[model_1, model_2])
inputs = tokenizer(["Hello, my dog is cute","Hello, my dog is cute"], return_tensors="pt")
inputs.pop('token_type_ids')
g = G.generate(**inputs,
           max_length=80,
           num_beams=3,
           length_penalty=0.8)
print(g)