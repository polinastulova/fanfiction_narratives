f = open("phantom_origin.txt", "r", encoding = "utf-8-sig")
phantom_original = f.read()
sample = phantom_original

import spacy

nlp = spacy.load('en_core_web_sm')

document = nlp(sample)

text_no_namedentities = []

ents = [e.text for e in document.ents]
#print(ents)

#for ent in document.ents:
 #   print(ent.text, ent.label_)

for ent in document.ents:
    label = ent.label_
    if label == "PERSON":
       sample = sample.replace(ent.text, "")
print(sample)


#for item in document:
#    if item.text in ents:
 #       i = ents.index(item.text)
#        label = document.ents[i].label_
 #       if label == "PERSON":
  #         pass
  #  else:
   #     text_no_namedentities.append(item.text)
#print(" ".join(text_no_namedentities))
