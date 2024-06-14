import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import heapq
import re
import pandas as pd
import joblib
import base64

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from gensim.models import phrases, word2vec, Word2Vec
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from sklearn.decomposition import PCA

from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')

def main():
    #Add Background
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
            unsafe_allow_html=True
        )

    add_bg_from_local('bg_pic.jpg')
    # End Background
    # Title Section
    st.title('Stock News Interpreter App')
    #End Title
    contents = ['Home', 'Guide', 'About']
    choice = st.sidebar.selectbox('Contents', contents)

    # functions
    def get_summary(text):
        word_freq = {}
        for word in nltk.word_tokenize(text):
            if word not in stopwords:
                if word not in word_freq.keys():
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
            # End of dictionary creation

        max_freq = max(word_freq.values())

        # Calculate weightage of each word
        for word in word_freq.keys():
            word_freq[word] = (word_freq[word] / max_freq)

        # Create dictionary to score each sentence
        sentence_scores = {}
        sentence_list = text.split("  ")
        for sent in sentence_list:
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_freq.keys():
                    if len(sent.split(' ')) < 30:
                        # takes sentences that are not interpreted yet i.e. first words of each sentence
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_freq[word]
                        else:
                            sentence_scores[sent] += word_freq[word]

        # Extract top 5 sentences with the highest scores. 5 is used as an arbitrary value in the optimization stage
        summary_sentences = heapq.nlargest(5, sentence_scores, key=sentence_scores.get)

        summary = ' '.join(summary_sentences)
        return summary

    def extract_ngrams(data, num):
        n_grams = ngrams(nltk.word_tokenize(data), num)
        n_gram_ls = ['_'.join(grams) for grams in n_grams]
        return ' '.join(gr for gr in n_gram_ls)

    stock_syms = pd.read_csv('stock_syms.csv')
    stock_syms = list(stock_syms.iloc[:, 0])

    def prepare_data(input_text, stock_syms):
        sent_prep = re.sub(r'[^\w\s]', '', input_text)
        sent_prep = re.sub(r'[\d*]', '', input_text)
        sent_prep = re.sub(r'[\n]', '', input_text)

        custom_stop_words = []
        with open("customstopwords.txt", "r") as fin:  # Expand more stop words if required
            for line in fin.readlines():
                custom_stop_words.append(line.strip())

        cust_stop_words = text.ENGLISH_STOP_WORDS.union(custom_stop_words)
        cust_stop_words = list(cust_stop_words)
        sent_prep = [word for word in nltk.wordpunct_tokenize(sent_prep) if word.lower() not in cust_stop_words]
        lemmatizer = WordNetLemmatizer()
        sent_prep = [lemmatizer.lemmatize(word) for word in sent_prep]
        sent_prep = " ".join(word for word in sent_prep if word.lower() not in stock_syms)
        # Create ngrams in phrases
        ngram_prep = nltk.word_tokenize(extract_ngrams(sent_prep, 2))
        ngram_prep = [word.lower() for word in ngram_prep]
        summ = [" ".join(word for word in ngram_prep)]
        return summ, ngram_prep

    lda_model = joblib.load('lda_model.jl')
    lr_model = joblib.load('lr_model.jl')
    stopwords = nltk.corpus.stopwords.words('english')
    #End prep functions


    if choice == 'Home':
        st.subheader('Stock News Interpreter Form')
        #initialize output
        text_in = ''
        pre_text_in = ''
        predict_out = ''
        col1, col2 = st.columns(2)
        topic_terms = pd.DataFrame()
        with col1:
            with st.form(key='main_form', clear_on_submit=False):
                text_in = st.text_area('Article or Text to Analyze:')
                submit = st.form_submit_button(label='Submit')

        if submit:
            if text_in:
                text_to_list = text_in.split(' ')
                #Clear output
                predict_out = ''
                #Start prediction
                pre_text_in = text_in #Unformatted text input
                text_in = re.sub(r'[^\w\s]', ' ', text_in)
                test, test2 = prepare_data(text_in, stock_syms)
                tf_vectorizer = TfidfVectorizer()
                try:
                    tfvec_doc = tf_vectorizer.fit_transform(test)
                    lda_output = lda_model.fit_transform(tfvec_doc)  # returns eg. array([[0.91550219, 0.02111034, 0.02114677, 0.02111765, 0.02112305]])
                    #print(lda_output)
                    #Get the most relevant topic out of the lda prediction
                    most_rel_top = np.argmax(lda_output, axis=1)[0]
                    #Get top topics related to the one predicted
                    topics = ["Topic " + str(i) for i in range(lda_model.n_components)]
                    topics_df = pd.read_csv('lda_topics.csv')
                    #print(topics_df)
                    topic_cols = topics_df.columns[1:]
                    #Get dataframe of top 10 terms of topic
                    topic_terms = topics_df.loc[topics_df.iloc[:,0] == topics[most_rel_top], topic_cols]
                    myvar = pd.DataFrame(lda_output, columns=topics)
                    #Get sentiment
                    sid = SentimentIntensityAnalyzer()
                    polr = sid.polarity_scores(text_in)['compound']
                    myvar['polarity'] = polr
                    if len(myvar) > 0:
                        lr_output = lr_model.predict(myvar)
                        #print(lr_output)
                    if lr_output == 0:
                        predict_out = ':red[Negative market performance predicted.]'
                    elif lr_output == 1:
                        predict_out = ':green[Positive market performance predicted.]'
                except ValueError:
                    predict_out = 'Value error. \nPlease enter a longer piece of text.'

        with col2:
            with st.expander(label='Prediction', expanded=True):
                if not(text_in) and submit:
                    st.markdown('**:red[Empty input field. Please try again.]**')
                elif not(text_in) and not(submit):
                    st.text('Submit a piece of text to continue.')
                elif text_in and submit:
                    st.markdown('**Prediction for input:**')
                    st.text(f'{pre_text_in[:30]}...')
                    st.divider()
                    st.markdown(predict_out)
        if not(topic_terms.empty):
            topics_output = st.container()
            topics_output.subheader(f'This article is most related to topic {most_rel_top}.')
            topics_output.text('Below are the top 10 terms that decribe this topic:')
            topics_output.dataframe(topic_terms)
    elif choice == 'About':
        st.subheader("About this project")
        st.divider()
        about = st.container()
        about.text('This project was created to predict stock sentiments and market performance, as well\n'
                   'as predict the topic the article is most similar to. The inspiration for the project\n'
                   'was to leverage on technology to give investors a competitive edge in their \n'
                   'day-to-day trades. This application was intended to be used alongside other trading \n'
                   'strategies and not to be used as a standalone tool.')
    elif choice == 'Guide':
        st.subheader('User Guide')
        st.divider()
        video_file = open('userguide.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

if __name__ == '__main__':
    main()