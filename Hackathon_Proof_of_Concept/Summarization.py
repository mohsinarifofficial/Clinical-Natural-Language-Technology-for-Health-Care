import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest


punctuation=punctuation+"\n"

stop_words=list(STOP_WORDS)
#print(STOP_WORDS)
nlp=spacy.load("en_core_sci_md")
text=""""Clinical narratives represent the main form of 
communicatio nwithin health care,
providing a personalized account of patient history and assessments,
and offering rich information for clinical decision making.
Natural language processing (NLP) has repeatedly demonstrated
its feasibility to unlock evidence buried in clinical narratives.
Machine learning can facilitate rapid development
 of NLP tools by leveraging large amounts of text data."""

text=str(text)
doc=nlp(text)

tokens=[i.text for i in doc]
#step 1
#print(tokens)
word_freq={}
for i in doc:
    if i.text.lower()  not in stop_words:
        if i.text.lower() not in punctuation:
            if i.text.lower() not in word_freq:
                word_freq[i.text.lower()]=1
            else:
                word_freq[i.text.lower()]+=1
                #print(i.text)
#print(word_freq)                    
max_val=max(word_freq.values())

#print(max_val)
#normalization step 2
for i in word_freq.keys():
    word_freq[i]=word_freq[i]/max_val
#print(word_freq)    
#step 3
sent_token=[i for i in doc.sents]
sent_scoer={}
for i in sent_token:
    #print(i)
    for j in i:
        if j.text.lower() in word_freq.keys():
            if i not in sent_scoer:
                sent_scoer[i]=word_freq[j.text.lower()]
            else:
                sent_scoer[i]+=word_freq[j.text.lower()]
#print(sent_scoer)                  
#step 4 
#print(len(sent_token))
l=int(len(sent_token)*0.50)
#print(l)

summary=nlargest(l,sent_scoer,key=sent_scoer.get)
print(summary)






