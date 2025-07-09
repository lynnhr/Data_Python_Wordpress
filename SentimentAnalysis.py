73% of storage used … If you run out of space, you can't save to Drive or use Gmail. Get 100 GB of storage for $1.99 US$0 for 1 month.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk

#----------------------------------USING NLTK----------------------------------
df=pd.read_csv('galaxys24ultra.csv')
example=df['Description'][50]
# #SPLIT EACH WORD
# tokens=nltk.word_tokenize(example)
# #nltk.pos_tag(tokens) #to show their part of speech

#VADER does not account for relationships between words
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
#create our object
sia = SentimentIntensityAnalyzer()
#print(sia.polarity_scores('I am so happy!'))
res={}
for i, row in tqdm(df.iterrows(), total=len(df)):
     text=row['Description']
     author=row['Name']
     res[author]=sia.polarity_scores(text)

#check the accuracy

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Name'})
vaders = vaders.merge(df, how='left')
pd.DataFrame(vaders).to_csv('vaders2.csv')
#
#
# fig, axs=plt.subplots(1,3,figsize=(15,5))
# sns.barplot(data=vaders, x='Stars', y='pos', ax=axs[0])
# sns.barplot(data=vaders, x='Stars', y='neu', ax=axs[1])
# sns.barplot(data=vaders, x='Stars', y='neg', ax=axs[2])
# axs[0].set_title('Positive')
# axs[1].set_title('Neutral')
# axs[2].set_title('Negative')
# plt.show()

#----------------------------------Roberta----------------------------------
# from transformers import AutoTokenizer
# from transformers import AutoModelForSequenceClassification
# from scipy.special import softmax
#
# MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL)
#
# def polarity_scores_roberta(example):
#  encoded_text = tokenizer(example, return_tensors='pt') #1 and 0 that the model will understand
#  output=model(**encoded_text) #run the model on the encoded text
#  scores = output[0][0].detach().numpy()
#  scores=softmax(scores)
#  scores_dict={
#     'roberta_neg':scores[0],
#     'roberta_neutral':scores[1],
#     'roberta_pos':scores[2]
# }
#  return scores_dict
#
# res = {}
# for i, row in tqdm(df.iterrows(), total=len(df)):
#     try:
#         text=row['Description']
#         author=row['Name']
#         vader_result = sia.polarity_scores(text)
#         vader_result_rename = {}
#         for key, value in vader_result.items():
#             vader_result_rename[f"vader_{key}"] = value
#         roberta_result = polarity_scores_roberta(text)
#         both = {**vader_result_rename, **roberta_result}
#         res[author] = both
#     except RuntimeError:
#         print(f'Broke for id {author}')
#
# results_df = pd.DataFrame(res).T
# results_df = results_df.reset_index().rename(columns={'index': 'Name'})
# results_df = results_df.merge(df, how='left')
# pd.DataFrame(results_df).to_csv('result_df.csv')
#
# #----------------------------------Comparing results----------------------------------
# sns.pairplot(data=results_df,
#              vars=['vader_neg', 'vader_neu', 'vader_pos',
#                    'roberta_neg', 'roberta_neutral', 'roberta_pos'],
#              hue='Stars',
#              palette='tab10')
# plt.show()


