from sacrebleu.metrics import BLEU, CHRF, TER
import pandas as pd

data = pd.read_csv("./output_multilingual_individualistic_collectivist.csv")

refs = data["English_Prompt"].apply(lambda x: [x]).tolist() 
print(refs)
# print(type(refs))

sys = data["Translated"].tolist()
# print(sys) 

bleu = BLEU()

print(bleu.corpus_score(sys, refs))

print(bleu.get_signature())

chrf = CHRF()

print(chrf.corpus_score(sys, refs))
