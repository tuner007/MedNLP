
# coding: utf-8

# In[59]:


import pandas as pd
from sklearn.externals import joblib
import re
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
import operator
import numpy as np
import sklearn.feature_extraction.text as text
from sklearn import decomposition
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.decomposition import PCA
from numpy.linalg import norm


# In[60]:


def load_data():
    ## Intitializing data paths
    base_path = r'D:\ORGANIZATION\UCSD_Life\Work\4. Quarter-3\Subjects\MED 277\Project\DATA\\'
    data_file = base_path+"NOTEEVENTS.csv.gz"
    
    ## Loading data frames from CSV file
    #df = pd.read_csv(data_file, compression='gzip')
    #df = df[:10000]
    #joblib.dump(df,base_path+'data10.pkl')
    
    ## loading data frames from PKL memory
    df1 =  joblib.load(base_path+'data10.pkl')
    df = df1[:50]
    
    ## Filtering dataframe for "Discharge summaries" and "TEXT"
    df = df.loc[df['CATEGORY'] == 'Discharge summary'] #Extracting only discharge summaries
    df_text = df['TEXT']
    return df_text


# ## EXTRACT ALL THE TOPICS

# In[61]:


'''Method that processes the entire document string'''
def process_text(txt):
    txt1 = re.sub('[\n]'," ",txt)
    txt1 = re.sub('[^A-Za-z \.]+', '', txt1)
    
    return txt1


# In[62]:


'''Method that processes the document string not considering separate lines'''
def process(txt):
    txt1 = re.sub('[\n]'," ",txt)
    txt1 = re.sub('[^A-Za-z ]+', '', txt1)
    
    _wrds = txt1.split()
    stemmer = SnowballStemmer("english") ## May use porter stemmer
    wrds = [stemmer.stem(wrd) for wrd in _wrds]
    return wrds


# In[63]:


'''Method that processes raw string and gets a processes list containing lines'''
def get_processed_sentences(snt_txt):
    snt_list = []
    for line in snt_txt.split('.'):
        line = line.strip()
        if len(line.split()) >= 5:
            snt_list.append(line)
    return snt_list


# In[64]:


'''This method extracts topic from sentence'''
def extract_topic(str_arg, num_topics = 1, num_top_words = 3):
    vectorizer = text.CountVectorizer(input='content', analyzer='word', lowercase=True, stop_words='english')
    try:
        dtm = vectorizer.fit_transform(str_arg.split())
        vocab = np.array(vectorizer.get_feature_names())
    
        #clf = decomposition.NMF(n_components=num_topics, random_state=1) ## topic extraction
        clf = decomposition.LatentDirichletAllocation(n_components=num_topics, learning_method='online')
        clf.fit_transform(dtm)

        topic_words = []
        for topic in clf.components_:
            word_idx = np.argsort(topic)[::-1][0:num_top_words] ##[::-1] reverses the list
            topic_words.append([vocab[i] for i in word_idx])
        return topic_words
    except:
        return None


# In[65]:


'''This method extracts topics of each sentence and returns a list'''
def extract_topics_all(doc_string):
    #One entry per sentence in list
    doc_str = process_text(doc_string)
    doc_str = get_processed_sentences(doc_str)
    
    res = []
    for i in range (0, len(doc_str)):
        snd_str = doc_str[i].lower()
        #print("Sending ----------------------------",snd_str,"==========",len(snd_str))
        tmp_topic = extract_topic(snd_str, num_topics = 2, num_top_words = 1)
        for top in tmp_topic:
            for wrd in top:
                res.append(wrd)
    return res


# In[66]:


'''This function takes a dataframe and returns all the topics in the entire corpus'''
def extract_corpus_topics(arg_df):
    all_topics = set()
    cnt = 1
    for txt in arg_df:
        all_topics = all_topics.union(extract_topics_all(txt))
        print("Processed ",cnt," records")
        cnt += 1
    all_topics = list(all_topics)
    return all_topics


# ## GET A VECTORIZED REPRESENTATION OF ALL THE TOPICS

# In[67]:


'''data_set = words list per document.
    vocabulary = list of all the words present
    _vocab = dict of word counts for words in vocabulary'''
def get_vocab_wrd_map(df_text):
    data_set = []
    vocabulary = []
    _vocab = defaultdict(int)
    for i in range(0,df_text.size):
        txt = process(df_text[i])
        data_set.append(txt)

        for wrd in txt:
            _vocab[wrd] += 1

        vocabulary = vocabulary + txt
        vocabulary = list(set(vocabulary))

        if(i%100 == 0):
            print("%5d records processed"%(i))
    return data_set, vocabulary, _vocab


# In[68]:


'''vocab = return sorted list of most common words in vocabulary'''
def get_common_vocab(num_arg, vocab):
    vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
    vocab = vocab[:num_arg]
    return vocab


# In[69]:


'''Convert vocabulary and most common words to map for faster access'''
def get_vocab_map(vocabulary, vocab):
    vocab_map = {}
    for i in range(0,len(vocab)):
        vocab_map[vocab[i][0]] = i 
    
    vocabulary_map = {}
    for i in range(0,len(vocabulary)):
        vocabulary_map[vocabulary[i]] = i
    
    return vocabulary_map, vocab_map


# In[70]:


def get_embedding(word, data_set, vocab_map, wdw_size):
    embedding = [0]*len(vocab_map)
    for docs in data_set:
        for i in range(wdw_size, len(docs)-wdw_size):
            if docs[i] == word:
                for j in range(i-wdw_size, i-1):
                    if docs[j] in vocab_map:
                        embedding[vocab_map[docs[j]]] += 1
                for j in range(i+1, i+wdw_size):
                    if docs[j] in vocab_map:
                        embedding[vocab_map[docs[j]]] += 1
    total_words = sum(embedding)
    if total_words != 0:
        embedding[:] = [e/total_words for e in embedding]
    return embedding


# In[71]:


def get_embedding_all(all_topics, data_set, vocab_map, wdw_size):
    embeddings = []
    for i in range(0, len(all_topics)):
        embeddings.append(get_embedding(all_topics[i], data_set, vocab_map, wdw_size))
    return embeddings


# ## Get similarity function

# In[72]:


def cos_matrix_multiplication(matrix, vector):
    """
    Calculating pairwise cosine distance using matrix vector multiplication.
    """
    dotted = matrix.dot(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    matrix_vector_norms = np.multiply(matrix_norms, vector_norm)
    neighbors = np.divide(dotted, matrix_vector_norms)
    return neighbors


# In[73]:


def get_most_similar_topics(embd, embeddings, all_topics, num_wrd=10):
    sim_top = []
    cos_sim = cos_matrix_multiplication(np.array(embeddings), embd)
    #closest_match = cos_sim.argsort()[-num_wrd:][::-1] ## This sorts all matches in order
    
    ## This just takes 80% and above similar matches
    idx = list(np.where(cos_sim > 0.9)[0])
    val = list(cos_sim[np.where(cos_sim > 0.9)])
    closest_match, list2 = (list(t) for t in zip(*sorted(zip(idx, val), reverse=True)))
    closest_match = np.array(closest_match)
    
    for i in range(0, closest_match.shape[0]):
        sim_top.append(all_topics[closest_match[i]])
    return sim_top


# ## Topic Modelling

# In[74]:


def get_regex_match(regex, str_arg):
    srch = re.search(regex,str_arg)
    if srch is not None:
        return srch.group(0).strip()
    else:
        return "Not found"


# In[75]:


def extract(key,str_arg):
    if key == 'dob':
        return get_regex_match('Date of Birth:(.*)] ', str_arg)
    elif key == 'a_date':
        return get_regex_match('Admission Date:(.*)] ', str_arg)
    elif key == 'd_date':
        return get_regex_match('Discharge Date:(.*)]\n', str_arg)
    elif key == 'sex':
        return get_regex_match('Sex:(.*)\n', str_arg)
    elif key == 'service':
        return get_regex_match('Service:(.*)\n', str_arg)
    elif key == 'allergy':
        return get_regex_match('Allergies:(.*)\n(.*)\n', str_arg)
    elif key == 'attdng':
        return get_regex_match('Attending:(.*)]\n', str_arg)
    else:
        return "I Don't know"


# In[76]:


'''This method extracts topic from sentence'''
def extract_topic(str_arg, num_topics = 1, num_top_words = 3):
    vectorizer = text.CountVectorizer(input='content', analyzer='word', lowercase=True, stop_words='english')
    dtm = vectorizer.fit_transform(str_arg.split())
    vocab = np.array(vectorizer.get_feature_names())
    
    #clf = decomposition.NMF(n_components=num_topics, random_state=1) ## topic extraction
    clf = decomposition.LatentDirichletAllocation(n_components=num_topics, learning_method='online')
    clf.fit_transform(dtm)
    
    topic_words = []
    for topic in clf.components_:
        word_idx = np.argsort(topic)[::-1][0:num_top_words] ##[::-1] reverses the list
        topic_words.append([vocab[i] for i in word_idx])
    return topic_words


# In[77]:


'''This method extracts topics in a question'''
def extract_Q_topic(str_arg):
    try:
        return extract_topic(str_arg)
    except:
        return None
    ## TODO fix later for more comprehensive results


# In[78]:


def get_extract_map(key_wrd):
    ## A Stemmed mapping for simple extractions
    extract_map = {'birth':'dob', 'dob':'dob',
              'admiss':'a_date', 'discharg':'d_date',
              'sex':'sex', 'gender':'sex', 'servic':'service',
              'allergi':'allergy', 'attend':'attdng'}
    if key_wrd in extract_map.keys():
        return extract_map[key_wrd]
    else:
        return None


# In[79]:


'''Method that generates the answer for text extraction questions'''
def get_extracted_answer(topic_str, text):
    port = PorterStemmer()
    for i in range(0, len(topic_str)):
        rel_wrd = topic_str[i]
        for wrd in rel_wrd:
            key = get_extract_map(port.stem(wrd))
            if key is not None:
                return extract(key, text)
    return None


# In[80]:


'''This method extracts topics of each sentence and returns a list'''
def get_topic_mapping(doc_string):
    #One entry per sentence in list
    doc_str = process_text(doc_string)
    doc_str = get_processed_sentences(doc_str)
    
    res = defaultdict(list)
    for i in range (0, len(doc_str)):
        snd_str = doc_str[i].lower()
        #print("Sending ----------------------------",snd_str,"==========",len(snd_str))
        tmp_topic = extract_topic(snd_str, num_topics = 2, num_top_words = 1)
        for top in tmp_topic:
            for wrd in top:
                res[wrd].append(doc_str[i])
    return res


# In[81]:


def get_direct_answer(topic_str, topic_map):
    ## Maybe apply lemmatizer here
    for i in range(0, len(topic_str)):
        rel_wrd = topic_str[i]
        for wrd in rel_wrd:
            if wrd in topic_map.keys():
                return topic_map[wrd]
    return None


# In[82]:


def get_answer(topic, topic_map, embedding_short, all_topics, data_set, vocab_map, pca, wdw_size=5):
    ## Get most similar topics
    tpc_embedding = get_embedding(topic, data_set, vocab_map, wdw_size)
    tpc_embedding = pca.transform([tpc_embedding])
    sim_topics = get_most_similar_topics(tpc_embedding[0], embedding_short, all_topics, num_wrd = len(all_topics))
    for topic in sim_topics:
        if topic in topic_map.keys():
            return topic_map[topic]
    return None


# In[83]:


'''This function checks if the user input text is an instruction allowed in chatbot or not'''
def is_instruction_option(str_arg):
    if str_arg == "exit" or str_arg == "summary" or str_arg == "reveal":
        return True
    else:
        return False

def print_bot():
	print(r"          _ _ _")
	print(r"         | o o |")
	print(r"        \|  =  |/")
	print(r"         -------")
	print(r"         |||||||")
	print(r"         //   \\")
	
def print_caption():
	print(r"	||\\   ||  ||       ||= =||")
	print(r"	|| \\  ||  ||       ||= =||")
	print(r"	||  \\ ||  ||       ||")
	print(r"	||   \\||  ||_ _ _  ||")


# In[ ]:


if __name__ == "__main__":
    print("Loading data ...","\n")
    df_text = load_data()
    
    print("Getting Vocabulary ...")
    data_set, vocabulary, _vocab = get_vocab_wrd_map(df_text)
    
    print("Creating context ...")
    vocab = get_common_vocab(1000, _vocab)
    vocabulary_map, vocab_map = get_vocab_map(vocabulary, vocab)
    
    print("Learning topics ...")
    all_topics = extract_corpus_topics(df_text)
    
    print("Getting Embeddings")
    embeddings = get_embedding_all(all_topics, data_set, vocab_map, 5)
    
    pca = PCA(n_components=10)
    embedding_short = pca.fit_transform(embeddings)
    
    print_caption()
    print_bot()
    print("Bot:> I am online!")
    print("Bot:> Type \"exit\" to switch to end a patient's session")
    print("Bot:> Type \"summary\" to view patient's discharge summary")
    while(True):
        while(True):
            try:
                pid = int(input("Bot:> What is your Patient Id [0 to "+str(df_text.shape[0]-1)+"?]"))
            except:
                continue
            if pid < 0 or pid > df_text.shape[0]-1:
                print("Bot:> Patient Id out or range!")
                continue
            else:
                print("Bot:> Reading Discharge Summary for Patient Id: ",pid)
                break

        personal_topics = extract_topics_all(df_text[pid])
        topic_mapping = get_topic_mapping(df_text[pid])
        
        ques = "random starter"
        while(ques != "exit"):
            ## Read Question
            ques = input("Bot:> How can I help ?\nPerson:>")
            
            ## Check if it is an instructional question
            if is_instruction_option(ques):
                if ques == "summary":
                    print("Bot:> ================= Discharge Summary for Patient Id ",pid,"\n")
                    print(df_text[pid])
                elif ques == "reveal":
                    print(topic_mapping, topic_mapping.keys())
                continue
                
            ## Extract Question topic
            topic_q = extract_Q_topic(ques)
            if topic_q is None:
                print("Bot:> I am a specialized NLP bot, please as a more specific question for me!")
                continue
            ans = get_extracted_answer(topic_q, df_text[pid])
            if ans is not None:
                print("Bot:> ",ans)
            else:
                ans = get_direct_answer(topic_q, topic_mapping)
                if ans is not None:
                    print("Bot:> ",ans)
                else:
                    ans = get_answer(topic_q, topic_mapping, embedding_short, all_topics, data_set, vocab_map, pca, 5)
                    if ans is not None:
                        print("Bot:> ",ans)
                    else:
                        print("Bot:> Sorry but, I have no information on this topic!")
