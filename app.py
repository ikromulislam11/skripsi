from flask import Flask, render_template, redirect, request, url_for
from datetime import datetime
import twint
import nest_asyncio
nest_asyncio.apply()
import pandas as pd
from pandas import Series, DataFrame
from flask_mysqldb import MySQL
import pymysql
import os
import numpy as np
import re
import string
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import json
import openpyxl
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer


application = Flask(__name__)

application.config['MYSQL_HOST'] = 'sql6.freemysqlhosting.net'
application.config['MYSQL_USER'] = 'sql6422853'
application.config['MYSQL_PASSWORD'] = 'jejPy6DdFs'
application.config['MYSQL_DB'] = 'sql6422853'
mysql = MySQL(application)

@application.route('/')
def index():   
   return render_template('index.html')

@application.route('/crawling')
def upload():
   return render_template('crawling.html')

@application.route('/crawling', methods=['GET','POST'])
def crawling():
    try:
       c = twint.Config()
       c.Search = request.form.get('query')
       c.Limit = request.form.get('jumlah')
       c.Since = "2019-04-29 00:00:00"
       c.Pandas = True
       c.Pandas_clean = True
       hasil = twint.run.Search(c)
       df = twint.storage.panda.Tweets_df.drop_duplicates(subset='id')

       df1 = df[['id', 'username', 'tweet']]
       column_names=df1.columns.values
       row_data=list(df1.values.tolist())
       jumlah_data = len(df1.index)
       
       id_data = df['id'].str.encode('utf-8')
       username = df['username'].str.encode('utf-8')
       text = df['tweet'].str.encode('utf-8')
       
       for (a, b, c) in zip(id_data, username, text):
          cur = mysql.connection.cursor()
          cur.execute("INSERT IGNORE INTO crawling(id_tweet, username, text, waktu_crawling, label) VALUES (%s,%s,%s, curdate(), NULL)",(a, b, c,))
          mysql.connection.commit()

       tombol="""<button style="height: 30px; font-size: 14px;"><a href="lihat-data" style="color: white; text-decoration: none;">
              Pelabelan Manual</button>"""
       #cur.execute("SELECT * FROM crawling where waktu_crawling = curdate() order by nomor asc")
       #rv = cur.fetchall()
       #cur.close()
       return render_template('crawling.html', column_names=column_names, row_data=row_data, zip=zip, jumlah_data=jumlah_data, tombol=tombol)
    except:
       info = "Crawling gagal. Periksa koneksi internet"
       return render_template('crawling.html', info=info)

@application.route('/edit/<id_tweet>', methods=['GET','POST'])
def edit(id_tweet):
   con = pymysql.connect(host='sql6.freemysqlhosting.net',
           port = 3306,
           user='sql6422853',
           password='jejPy6DdFs',
           db='sql6422853',
           charset='utf8mb4')
   cur = mysql.connection.cursor()
   cur.execute('SELECT * FROM crawling WHERE id_tweet=%s', (id_tweet,))
   mysql.connection.commit()
   data = cur.fetchone()
   if request.method == 'POST':
      id_tweet = request.form['id_tweet']
      label = request.form['label']
      cur = mysql.connection.cursor()
      cur.execute('UPDATE crawling SET label=%s WHERE id_tweet=%s', (label, id_tweet,))
      mysql.connection.commit()
      cur.close
      return redirect(url_for('lihat_data'))
   else:
      cur.close()
      return render_template('update-data.html', data=data)

@application.route('/lihat-data')
def lihat_data():
    con = pymysql.connect(host='sql6.freemysqlhosting.net',
           port = 3306,
           user='sql6422853',
           password='jejPy6DdFs',
           db='sql6422853',
           charset='utf8mb4')
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM crawling")
    mysql.connection.commit()
    rv = cur.fetchall()
    cur.execute("SELECT COUNT(*) FROM crawling")
    jumlah_data = cur.fetchone()
    cur.close()
    return render_template('lihat-data.html', tables = rv, jumlah_data=jumlah_data)

@application.route('/lihat-data/<id_tweet>', methods=['GET','POST'])
def hapus_data(id_tweet):
    con = pymysql.connect(host='sql6.freemysqlhosting.net',
           port = 3306,
           user='sql6422853',
           password='jejPy6DdFs',
           db='sql6422853',
           charset='utf8mb4')
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM crawling where id_tweet = %s", (id_tweet,))
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('lihat_data'))

@application.route('/preprocessing-data')
def preprocessing():
   return render_template('preprocessing.html')

@application.route('/data-preprocessing')
def data_preprocessing():
    con = pymysql.connect(host='sql6.freemysqlhosting.net',
           port = 3306,
           user='sql6422853',
           password='jejPy6DdFs',
           db='sql6422853',
           charset='utf8mb4')
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM preprocessing")
    mysql.connection.commit()
    rv = cur.fetchall()
    cur.execute("SELECT COUNT(*) FROM preprocessing")
    jumlah_data = cur.fetchone()
    cur.close()
    return render_template('data-preprocessing.html', tables=rv, jumlah_data=jumlah_data)

@application.route('/data-preprocessing', methods=['GET','POST'])
def hapus_preprocessing():
    con = pymysql.connect(host='sql6.freemysqlhosting.net',
           port = 3306,
           user='sql6422853',
           password='jejPy6DdFs',
           db='sql6422853',
           charset='utf8mb4')
    try:
       cur = mysql.connection.cursor()
       cur.execute("Delete FROM preprocessing")
       mysql.connection.commit
       cur.execute("SELECT * FROM preprocessing")
       mysql.connection.commit()
       rv = cur.fetchall()
       cur.execute("SELECT COUNT(*) FROM preprocessing")
       jumlah_data = cur.fetchone()
       cur.close()
       return render_template('data-preprocessing.html', tables=rv, jumlah_data=jumlah_data)
    except: # work on python 3.x
       info = 'Gagal hapus data : Terdapat data yang menjadi foreign key di tabel klasifikasi'
       cur = mysql.connection.cursor()
       cur.execute("SELECT * FROM preprocessing")
       mysql.connection.commit()
       rv = cur.fetchall()
       cur.execute("SELECT COUNT(*) FROM preprocessing")
       jumlah_data = cur.fetchone()
       cur.close()
       return render_template('data-preprocessing.html', tables=rv, jumlah_data=jumlah_data, info=info)

@application.route('/data-slangwords')
def slangword():
   #with open("D:\FD\skripsi\slangword.txt") as file:
      #file = file.read().splitlines()
   file = eval(open("slangword.txt").read())
   return render_template('slangwords.html', data=file)

@application.route('/data-stopwords')
def stopword():
   with open("stopword.txt") as file:
      file = file.read().splitlines()
   return render_template('stopwords.html', data=file)

@application.route('/update-slangwords', methods=['GET','POST'])
def update_slangword():
   if request.method == 'POST':
      slang = request.form['slangword']
      katabaku = request.form['katabaku']

      file = eval(open("slangword.txt").read())
      file[slang] = katabaku
      f = open("slangword.txt","w")
      f.write( str(file) )
      f.close()
      return redirect(url_for('slangword'))
   
   else:  
      return render_template('update-slangwords.html')

@application.route('/update-stopwords', methods=['GET','POST'])
def update_stopword():
   if request.method == 'POST':
      stop = request.form['form_stopword']

      with open("stopword.txt", "a") as f:
          f.write(", "+ "'"+stop+"'")

      return redirect(url_for('stopword'))
   
   else:   
      return render_template('update-stopwords.html')

@application.route('/preprocessing-data', methods=['GET', 'POST'])
def preprocess():
   con = pymysql.connect(host='sql6.freemysqlhosting.net',
           port=3306,
           user='sql6422853',
           password='jejPy6DdFs',
           db='sql6422853',
           charset='utf8mb4')
   ambil_data = pd.read_sql_query("SELECT * FROM crawling where label is not null", con)
   df = pd.DataFrame(ambil_data)
   def remove(tweet):
       #tweet=tweet.lower()
       tweet=re.sub(r'http\S+','',tweet)
       
       #hapus @username
       tweet=re.sub('@[^\s]+','',tweet)
       
       #hapus #tagger 
       tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
       
       #hapus tanda baca
       tweet=hapus_tanda(tweet)
       
       #hapus angka dan angka yang berada dalam string 
       tweet=re.sub(r'\w*\d\w*', '',tweet).strip()
       
       # remove old style retweet text "RT"
       tweet = re.sub(r'^RT[\s]+', '', tweet)
    
       # remove hyperlinks
       tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
       
       # remove hashtags
       # only removing the hash # sign from the word
       tweet = re.sub(r'#', '', tweet)
       
       #remove coma
       tweet = re.sub(r',','',tweet)
       
       #remove angka
       tweet = re.sub('[0-9]+', '', tweet)
       return tweet
   def hapus_tanda(tweet): 
       tanda_baca = set(string.punctuation)
       tweet = ''.join(ch for ch in tweet if ch not in tanda_baca)
       return tweet
   df['text'] = df['text'].apply(lambda x: remove(x))
   df.sort_values('text', inplace = True)
   df.drop_duplicates(subset ='text', keep = 'first', inplace = True)
   
   # removes pattern in the input text
   def remove_pattern(input_txt, pattern):
       r = re.findall(pattern, input_txt)
       for word in r:
           input_txt = re.sub(word, "", input_txt)
       return input_txt
   # remove special characters, numbers and punctuations
   df['text'] = df['text'].str.replace("[^a-zA-Z#]", " ")
   df['text'] = df.text.str.lower()
   
   def word_tokenize_wrapper(tweet):
      return word_tokenize(tweet)
   df['text'] = df['text'].apply(word_tokenize_wrapper)

   def convertToSlangword(tweet):
       kamus_slangword = eval(open("slangword.txt").read()) # Membuka dictionary slangword
       pattern = re.compile(r'\b( ' + '|'.join (kamus_slangword.keys())+r')\b') # Search pola kata (contoh kpn -> kapan)
       content = []
       for kata in tweet:
           filteredSlang = pattern.sub(lambda x: kamus_slangword[x.group()],kata) # Replace slangword berdasarkan pola review yg telah ditentukan
           content.append(filteredSlang.lower())
       review = content
       return review
   df['text'] = df['text'].apply(convertToSlangword)

   factory = StemmerFactory()
   stemmer = factory.create_stemmer()
   # stemmed
   def stemmed_wrapper(term):
       return stemmer.stem(term)
   term_dict = {}

   for document in df['text']:
       for term in document:
           if term not in term_dict:
               term_dict[term] = ' '

   for term in term_dict:
       term_dict[term] = stemmed_wrapper(term)
       
   # apply stemmed term to dataframe
   def get_stemmed_term(document):
       return [term_dict[term] for term in document]
   df['text'] = df['text'].apply(get_stemmed_term)

   kamus_stopwords = eval(open("stopword.txt").read())
   
   #remove stopword pada list token
   def stopwords_removal(words):
       return [word for word in words if word not in kamus_stopwords]
   df['text'] = df['text'].apply(stopwords_removal)
   
   text = df['text'].astype(str)
   id_data = df['id_tweet'].str.encode('utf-8')
   label = df['label'].str.encode('utf-8')

   for a,b,c in zip(id_data,text,label):
      cur = mysql.connection.cursor()
      cur.execute("UPDATE preprocessing SET label_pre = %s where id_tweet = %s", (c,a,))
      mysql.connection.commit()
      cur.execute("INSERT IGNORE INTO preprocessing(id_tweet, text_pre, label_pre) VALUES(%s,%s,%s)", (a,b,c,))
      mysql.connection.commit()
      cur.close()
   notif = 'Proses Preprocessing Selesai'
      
   return render_template('preprocessing.html', notif = notif)

@application.route('/halaman-tfidf')
def halaman_tfidf():
   return render_template('halaman-tfidf.html')

@application.route('/halaman-tfidf', methods=['GET','POST'])
def tfidf():
   con = pymysql.connect(host='sql6.freemysqlhosting.net',
           port=3306,
           user='sql6422853',
           password='jejPy6DdFs',
           db='sql6422853',
           charset='utf8mb4')
   ambil_data = pd.read_sql_query("SELECT text_pre FROM preprocessing LIMIT 1250", con)
   df = pd.DataFrame(ambil_data)
   
   vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=False, norm=None)
   vectors = vectorizer.fit_transform(df['text_pre'])
   feature_names = vectorizer.get_feature_names()
   dense = vectors.todense()
   denselist = dense.tolist()
   df = pd.DataFrame(denselist, columns=feature_names, index=df['text_pre'])
   file = df.to_excel("tfidf.xlsx")
   jumlah = len(df.columns)
   notif = 'Hasil TFIDF berhasil disimpan dalam file excel dengan nama tfidf dengan jumlah fitur = '+str(jumlah)

   klasifikasi = """<button><a href="klasifikasi" style="color: white; text-decoration: none;">Klasifikasi</button>"""
   
   return render_template('halaman-tfidf.html', notif = notif, klasifikasi=klasifikasi)

@application.route('/klasifikasi')
def halamanklasifikasi():
   return render_template('klasifikasi.html')
   
@application.route('/klasifikasi', methods=['GET','POST'])
def klasifikasi():
   con = pymysql.connect(host='sql6.freemysqlhosting.net',
           port = 3306,
           user='sql6422853',
           password='jejPy6DdFs',
           db='sql6422853',
           charset='utf8mb4')
   ambil_data = pd.read_sql_query("SELECT * FROM preprocessing LIMIT 1250", con)
   df = pd.DataFrame(ambil_data)
   model = GaussianNB()
   vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=False, norm=None)
   data_vector = vectorizer.fit_transform(df['text_pre']).toarray()

   X_train, X_test, y_train, y_test = train_test_split(data_vector, df['label_pre'], test_size=0.1, random_state=0)
   model.fit(X_train,y_train)
   y_preds = model.predict(X_test)

   text = df['text_pre'].astype(str)
   id_data = df['id_tweet'].str.encode('utf-8')
   label = df['label_pre'].str.encode('utf-8')
   label_baru = y_preds
   
   for a,b,c,d in zip(id_data,text,label, label_baru):
      cur = mysql.connection.cursor()
      cur.execute("INSERT IGNORE INTO klasifikasi(id_tweet_kla, text_kla, label_kla, label_baru) VALUES(%s,%s,%s,%s)", (a,b,c,d,))
      mysql.connection.commit()
   
   cur.execute("SELECT * FROM klasifikasi")
   hasil = cur.fetchall()
   cur.close()

   tombol = """<button style="margin-top: 15px"><a href="pengujian-model" style="color: white; text-decoration: none; font-size: 15px;">
            Pengujian Model</button>"""
   
   return render_template('klasifikasi.html', tables = hasil, tombol=tombol)

@application.route('/pengujian-model', methods=['GET','POST'])
def hitung_akurasi():
   con = pymysql.connect(host='sql6.freemysqlhosting.net',
           port = 3306,
           user='sql6422853',
           password='jejPy6DdFs',
           db='sql6422853',
           charset='utf8mb4')
   ambil_data = pd.read_sql_query("SELECT * FROM preprocessing LIMIT 1250", con)
   df = pd.DataFrame(ambil_data)
   model = GaussianNB()
   vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=False, norm=None)
   data_vector = vectorizer.fit_transform(df['text_pre']).toarray()
   
   X_train, X_test, y_train, y_test = train_test_split(data_vector, df['label_pre'], test_size=0.1, random_state=0)
   model.fit(X_train,y_train)
   y_preds = model.predict(X_test)
   
   lb = LabelBinarizer()
   y_train = np.array([number[0] for number in lb.fit_transform(df['label_pre'])])
   
      
   akurasi = cross_val_score(model, data_vector, y_train, cv=10)
   akurasibulat = np.around(akurasi, 2)
   akurasipersen = akurasibulat*100
   meanakurasi = np.mean(akurasipersen)

   precision = cross_val_score(model, data_vector, y_train, cv=10, scoring='precision')
   precisionbulat = np.around(precision, 2)
   precisionpersen = precisionbulat*100
   meanprecision = np.mean(precisionpersen)
   
   recall = cross_val_score(model, data_vector, y_train, cv=10, scoring='recall')
   recallbulat = np.around(recall, 2)
   recallpersen = recallbulat*100
   meanrecall = np.mean(recallpersen)
   
   return render_template('pengujian.html', akurasi = akurasi, akurasipersen = akurasipersen, recallpersen = recallpersen, precisionpersen = precisionpersen,
                          meanakurasi = meanakurasi, meanprecision = meanprecision, meanrecall = meanrecall)

@application.route('/implementasi-model')
def halamanimplementasi():
   return render_template('implementasi.html')

@application.route('/implementasi-model', methods=['GET','POST'])
def implementasi():
   c = twint.Config()
   c.Search = request.form.get('query')
   c.Limit = request.form.get('jumlah')
   c.Since = "2017-01-01 00:00:00"
   c.Pandas = True
   c.Pandas_clean = True
   hasil = twint.run.Search(c)
   df = twint.storage.panda.Tweets_df.drop_duplicates(subset='id')

   id_data = df['id'].str.encode('utf-8')
   username = df['username'].str.encode('utf-8')
   text = df['tweet'].str.encode('utf-8')
    
   for (a, b, c) in zip(id_data, username, text):
      cur = mysql.connection.cursor()
      cur.execute("INSERT IGNORE INTO implementasi(id_tweet_imp, username, text_imp, waktu_imp, label_imp) VALUES (%s,%s,%s, curdate(), NULL)",(a, b, c,))
      mysql.connection.commit()

   con = pymysql.connect(host='sql6.freemysqlhosting.net',
           port = 3306,
           user='sql6422853',
           password='jejPy6DdFs',
           db='sql6422853',
           charset='utf8mb4')
   
   ambil_data = pd.read_sql_query("SELECT * FROM preprocessing LIMIT 1250", con)
   ambil_data2 = pd.read_sql_query("SELECT * FROM implementasi", con)
   
   df_train = pd.DataFrame(ambil_data)
   df_test = pd.DataFrame(ambil_data2)
   
   def remove(tweet):
       #tweet=tweet.lower()
       tweet=re.sub(r'http\S+','',tweet)
       
       #hapus @username
       tweet=re.sub('@[^\s]+','',tweet)
       
       #hapus #tagger 
       tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
       
       #hapus tanda baca
       tweet=hapus_tanda(tweet)
       
       #hapus angka dan angka yang berada dalam string 
       tweet=re.sub(r'\w*\d\w*', '',tweet).strip()
       
       # remove old style retweet text "RT"
       tweet = re.sub(r'^RT[\s]+', '', tweet)
    
       # remove hyperlinks
       tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
       
       # remove hashtags
       # only removing the hash # sign from the word
       tweet = re.sub(r'#', '', tweet)
       
       #remove coma
       tweet = re.sub(r',','',tweet)
       
       #remove angka
       tweet = re.sub('[0-9]+', '', tweet)
       return tweet
   def hapus_tanda(tweet): 
       tanda_baca = set(string.punctuation)
       tweet = ''.join(ch for ch in tweet if ch not in tanda_baca)
       return tweet
   df_test['text_imp'] = df_test['text_imp'].apply(lambda x: remove(x))
   df_test.sort_values('text_imp', inplace = True)
   df_test.drop_duplicates(subset ='text_imp', keep = 'first', inplace = True)
   
   # removes pattern in the input text
   def remove_pattern(input_txt, pattern):
       r = re.findall(pattern, input_txt)
       for word in r:
           input_txt = re.sub(word, "", input_txt)
       return input_txt
   # remove special characters, numbers and punctuations
   df_test['text_imp'] = df_test['text_imp'].str.replace("[^a-zA-Z#]", " ")
   df_test['text_imp'] = df_test.text_imp.str.lower()
   
   def word_tokenize_wrapper(tweet):
      return word_tokenize(tweet)
   df_test['text_imp'] = df_test['text_imp'].apply(word_tokenize_wrapper)

   def convertToSlangword(tweet):
       kamus_slangword = eval(open("slangword.txt").read()) # Membuka dictionary slangword
       pattern = re.compile(r'\b( ' + '|'.join (kamus_slangword.keys())+r')\b') # Search pola kata (contoh kpn -> kapan)
       content = []
       for kata in tweet:
           filteredSlang = pattern.sub(lambda x: kamus_slangword[x.group()],kata) # Replace slangword berdasarkan pola review yg telah ditentukan
           content.append(filteredSlang.lower())
       review = content
       return review
   df_test['text_imp'] = df_test['text_imp'].apply(convertToSlangword)

   factory = StemmerFactory()
   stemmer = factory.create_stemmer()
   # stemmed
   def stemmed_wrapper(term):
       return stemmer.stem(term)
   term_dict = {}

   for document in df_test['text_imp']:
       for term in document:
           if term not in term_dict:
               term_dict[term] = ' '

   for term in term_dict:
       term_dict[term] = stemmed_wrapper(term)
       
   # apply stemmed term to dataframe
   def get_stemmed_term(document):
       return [term_dict[term] for term in document]
   df_test['text_imp'] = df_test['text_imp'].apply(get_stemmed_term)

   kamus_stopwords = eval(open("stopword.txt").read())
   
   #remove stopword pada list token
   def stopwords_removal(words):
       return [word for word in words if word not in kamus_stopwords]
   df_test['text_imp'] = df_test['text_imp'].apply(stopwords_removal)
   df_test['text_imp'].dropna()

   df_test['text_imp'] = df_test['text_imp'].astype(str)
   
   model = GaussianNB()
   
   vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=False, norm=None)
   data_vector = vectorizer.fit_transform(df_train['text_pre']).toarray()
   data_vector2 = vectorizer.transform(df_test['text_imp']).toarray()
   
   X_train = data_vector
   X_test = data_vector2
   y_train = df_train['label_pre']
   y_test = df_test['label_imp']

   model.fit(X_train,y_train)
   y_preds = model.predict(X_test)

   id_data_imp = df_test['id_tweet_imp'].str.encode('utf-8')

   for label, id_data in zip(y_preds, id_data_imp):
      cur = mysql.connection.cursor()
      cur.execute("UPDATE implementasi SET label_imp = %s where id_tweet_imp = %s",(label, id_data,))
      mysql.connection.commit()
      
   cur.execute("DELETE FROM implementasi where label_imp is NULL")
   mysql.connection.commit()
   cur.execute("SELECT COUNT(*) FROM implementasi")
   jumlah_data = cur.fetchone()
   mysql.connection.commit()
   cur.execute("SELECT * FROM implementasi")
   hasil = cur.fetchall()
   cur.close()

   return render_template('implementasi.html', tables = hasil, jumlah_data=jumlah_data)

port = int(os.environ.get("PORT", 5000))
if __name__ == '__main__':
   application.run(debug=True)
