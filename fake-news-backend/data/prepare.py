#The python pandas library used for data manipulation and analysis
import pandas as pd
#Tokenizer to convert text into IDs
from transformers import BertTokenizer

from transformers import BertForSequenceClassification,Trainer,TrainingArguments

#Library to serilaise objects
import pickle
#Initialising the tokenizer which is pretrained to know the exact vocab
#and tokenization rules for the BERT model
tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")#"uncased" so ignores capitalization

model=BertForSequenceClassification.from_pretrained("bert-base-uncased")

def load_and_prepare(file_path):
    df=pd.read_csv(file_path)#CSV filepath read into pandas DataFrame (df)
    #The text in the data is labelled 0 if FAKE and 1 if REAL, which is loaded in df
    texts=df["text"].tolist()#Extracts the column of texts and makes it python list of strings
    labels=df["label"].tolist()##Extracts the column of labels and makes it python list of strings

#BERT tokenizer applied to every texts in the list
#   -Truncation:cuts off long texts if they exceed max length supported by BERT(512 tokens)
#   -Padding:adds padding tokens so all sequences have same length
    encodings=tokenizer(texts,truncation=True,padding=True)
    #encodings is the dictionary containing:
    #         -input_ids:tokenized text(word IDs)
    #         -attention_mask:mask telling BERT which tokens are padding 

    return encodings,labels

#Save the encodings in the training model but thats not done in real time ,the input is encoded
#and fed to the trained model for prediction
def save_encodings(encodings,labels,save_path):
    with open(save_path,"wb") as f:#file opened for writing in binary mode where objects are saved
        #pickle.dump:to serialize python objects turn them into byte stream and saves them into file
        pickle.dump((encodings,labels),f)#encodings and labels bundled into a tuple,stored together and saved into a file



#Load and preparing three datasets:
train_enc,train_labels=load_and_prepare("fake-news-backend/data/train.csv")#training data
val_enc,val_labels=load_and_prepare("fake-news-backend/data/val.csv")#validation data
test_enc,test_labels=load_and_prepare("fake-news-backend/data/test.csv")#Test data

save_encodings(train_enc, train_labels, "fake-news-backend/data/train_enc.pkl")
save_encodings(val_enc, val_labels, "fake-news-backend/data/val_enc.pkl")
save_encodings(test_enc, test_labels, "fake-news-backend/data/test_enc.pkl")


model.save_pretrained("./fake-news-backend/saved_model")

tokenizer.save_pretrained("./fake-news-backend/saved_model")
#To load the saved info:
# import pickle

# with open("data/train_enc.pkl", "rb") as f:
#     train_enc, train_labels = pickle.load(f)

#TODO   :Add datasets from kaggle into the data folders .
# Each csv file has texts and labels(Which denotes whether the facts written in texts arr fake or real)




