import nltk
import re
import numpy as np
from nltk.translate import IBMModel1,AlignedSent
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
# nltk.download('stopwords')
"""
**********EXAMPLE BASED MACHINE TRANSLATION SYSTEM 'EBMT'******************************
Translation in three steps: matching, alignment and recombination
Matching: finds the example or set of examples from the bitext which most closely match the source-language string to be translated.4
Alignment: extracts the source–target translation equivalents from the retrieved examples of the matching step.
Recombination: produces the final translation by combining the target translations of the relevant subsentential fragments.
"""
############################## PREPROCESSING SECTION ###################################
replacement_patterns = [
(r'won\'t', 'will not'),
(r'can\'t', 'can not'),
(r'i\'m', 'i am'),
(r'ain\'t', 'is not'),
(r'(\w+)\'ll', '\g<1> will'),
(r'(\w+)n\'t', '\g<1> not'),
(r'(\w+)\'ve', '\g<1> have'),
(r'(\w+)\'s', '\g<1> is'),
(r'(\w+)\'re', '\g<1> are'),
(r'(\w+)\'d', '\g<1> would')
]
# stop words for each languages
en_Stp_wrd= set(stopwords.words('english'))
tr_Stp_wrd= set(stopwords.words('turkish'))
az_Stp_wrd= set(stopwords.words('azerbaijani'))
stpwrds= en_Stp_wrd | tr_Stp_wrd | az_Stp_wrd #union all to have a big set of all the stop wprds

# for contraction removals
class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s

def preprocessing(files):
    replacer = RegexpReplacer()
    for file in files:
        with open(file,'r',encoding='utf-8') as f:
            text= f.read()
        text = re.sub(r'[()\[\]{}.,;!?\\-]', '', text) #remove any punctutaions and unwanted characters
        text= text.replace('[','')
        text= text.replace(']','')
        text=text.replace('.','')
        text=text.replace(',','')
        text = text.strip() # spaces 
        text=text.lower() # lower case
        text=replacer.replace(text) # apply the stemming
        with open(file,'w',encoding='utf-8') as f1:
            f1.write(text) # save changes

# to remove duplicates from  a list of strings
def remove_duplicates(string_array):
    unique_strings = []
    duplicates = []
    for string in string_array:
        words = string.split()  # Split the string into words
        unique_words = []
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
            else:
                duplicates.append(word)
        unique_string = ' '.join(unique_words)
        if unique_string not in unique_strings:
            unique_strings.append(unique_string)
    
    return unique_strings

##############################################################################################################################################################################
# creates a parallel corpus for 2 languages given its files
def make_par_corpus(src_lang):
    with open (src_lang ,'r', encoding='utf-8')as f :
        src= f.readlines()
    with open('Turkish.txt' ,'r' ,encoding='utf-8') as f1:
        tk=f1.readlines()
    src = [s.strip() for s in src]
    tk = [t.strip() for t in tk]
    corpus = list(zip(src,tk))
    return corpus

def matching(input_sentence, src_lang):
    target_lang = 'Turkish.txt'  # fixed
    with open(src_lang, 'r', encoding='utf-8') as f1:
        src_sentences = f1.readlines()
    with open(target_lang, 'r', encoding='utf-8') as f2:
        turkish_sentences = f2.readlines()

    # Use the TF-IDF vectorizer to convert sentences to vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    sentence_vectors = vectorizer.fit_transform(src_sentences + [input_sentence])

    # Calculate the Cosine Similarity between the input sentence and all the other sentences
    input_vector = sentence_vectors[-1]
    similarity_scores = np.dot(sentence_vectors[:-1], input_vector.T).toarray().flatten()

    # Sort the similarity scores and get the indices of the top 4 sentences only
    top_indices = np.argsort(similarity_scores)[-5:]

    # Retrieve the top 4 similar sentences and their Turkish translations
    similar_sentences = [src_sentences[i] for i in top_indices]
    turkish_sentences = [turkish_sentences[i] for i in top_indices]
    turkish_tokens = [nltk.word_tokenize(s) for s in turkish_sentences]
    similar_tokens = [nltk.word_tokenize(s) for s in similar_sentences]
    parallel_sentences = list(zip(similar_tokens, turkish_tokens))
    
    print("similar sentences in the corpus: \n",parallel_sentences)
    return parallel_sentences  # return the top 4 similar source sentences with their translation equivalents



# IBM model for checking the alignment of words
def ibm1(src):
    with open(src, 'r',encoding='utf-8') as f1:
        src_sentences = f1.readlines()
    with open('Turkish.txt', 'r',encoding='utf-8') as f2:
        turkish_sentences = f2.readlines()
    #extra step step to remove spaces    
    src_sentences = [s.strip() for s in src_sentences]
    turkish_sentences = [t.strip() for t in turkish_sentences]
    # word tokenize all sentences
    src_tokenized=[nltk.word_tokenize(s) for s in src_sentences]
    tgt_tokenized =[nltk.word_tokenize(s) for s in turkish_sentences]
    # make a parallel sentences corpus but the sentences are tokenized
    parallel_sentences = list(zip(src_tokenized,tgt_tokenized))
    # Align all sentnences to train the IBM model
    bitext = [(AlignedSent(s,t)) for s,t in parallel_sentences]
    #train the model
    ibm1= IBMModel1(bitext,15)
    return ibm1 # return  the model

#to extracts the source–target translation equivalents from the retrieved examples of the matching step. 
def alignment(src,simialr_sentences,input_text):
    ibm=ibm1(src) # create the model
    turkish_translations= [word for sents in (pair[1] for pair in simialr_sentences) for word in sents] # extract all the turkish words from the similar sentences as the input
    # print(turkish_translations)
    text= nltk.word_tokenize(input_text)# tokenize the input
    equivlants=[] # it will hold pairs of the best match in turkish for each word in the input
    for s in text:
        all_probs= [(ibm.translation_table[s][w]) for w in turkish_translations] # find all the probabilites of each word in the input sentence with ALL tuekish words in the similar sentences
        for t in turkish_translations:
            # if s in stpwrds: # if a word is a stop word, ignore it
            #     continue
            prob=(ibm.translation_table[s][t]) #check probabilty of the word and a turkish word
            if prob == max(all_probs): # if its the highest probabilty , put it in pair so it acts like a dictionary
                equivlants.append((s,t))
                # print(f't({s}|{t} = {prob}')
                break
    return equivlants
    
# this function should produces the final translation by combining the target translations of the relevant subsentential fragments 
def combination(equivlants, input):
    combined_translation=''
    text = nltk.word_tokenize(input) #tokenize the input
    for w in text:
        for s,t  in equivlants:# check from the equivlants pairs we got from the alignment function
            if w == s:  # if the word in input exists in the equiv pairs, get the translated word
                combined_translation+=" "
                combined_translation += t
                combined_translation+=" "
    return combined_translation # return the sentence

def translate(input, src_lang): # Combine all 3 steps to produce the translation 
    flag=True # to check if the input already exist in the corpus
    corpus= make_par_corpus(src_lang) 
    for pair  in corpus:
        if input  == pair[0]: # if input sentence already exists, print its translation directly
            print("\ninput sentence: \n",input)
            print("\ntranslation: \n",pair[1])
            flag=False #set flag to false to disallow rest of code to run
            break
    if flag:
        similar_sentences = matching(input,src_lang)
        align=alignment(src_lang,similar_sentences,input)
        translation = combination(align,input)
        str= remove_duplicates(nltk.word_tokenize(translation))
        translation=" ".join(str)
        print("\ninput sentence: \n",input)
        print("\ntranslation: \n",translation)

########################## MAIN  ################################
files=['English.txt','Turkish.txt','Azeri.txt']
preprocessing(files) # preprocess all the files
translation= input('enter translation option:\n1)English-Turkish\n2)Azerbjiani-Turkish\n')

while(not(translation == '2' or translation == '1')):
    translation=input("input 1 or 2 only\n\nenter translation option:\n1)English-Turkish\n2)Azerbjiani-Turkish\n")
text = input('Enter a sentence to translate: \n')
while( len(text.split()) < 10): # <10
    text=input("Enter a sentence of atleast 10 words:\n")

if translation == '1':#For English-Turkish
    translate(text, files[0])
if translation =='2':
    translate(text,files[2])#For Azerbjiani-Turkish  

