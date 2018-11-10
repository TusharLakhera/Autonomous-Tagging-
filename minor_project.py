#importing the libraries
import pandas as shortpanda
import numpy as shortnum
import re
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import datetime
import unicodedata
import inflect
from collections import Counter
from bs4 import BeautifulSoup
#reading CSV Data
QuestionData =shortpanda.read_csv('Questions.csv',encoding='latin-1')
TagsData=shortpanda.read_csv('Tags.csv')
#creating a list of Tags from TagsData
Taglist=TagsData.Tag.tolist()
#creating a list of Tags from TagsData
SortedTaglist=Counter(Taglist)
print(SortedTaglist)  #analyzing whether the Data is sorted or not
list(QuestionData)
CreationDateList=QuestionData.CreationDate.tolist()
print(CreationDateList)

#finding the week number
Week= datetime.date(2014,4,17).isocalendar()[1]
Week= datetime.date(2012,6,12).isocalendar()[1]


#counting the number of most frequent word in Title Data
Titlelist=QuestionData.Title.tolist()

print(Titlelist)


with open('TempData.txt') as titlefile:
    TitleRead=titlefile.read()
    Words = re.findall(r'\w+', TitleRead)
   
    WordCount = Counter(Words)
    

#Cleaning Data by removing HTML Tags ,Links and Code Snippets
   
    Bodylist=QuestionData.Body.tolist() # list of unproceesed question data
#Using BeautifulSoup for Noise Removal
    
#for cleaning HTML headers
with open('SampleBodyData.txt') as Datafile:
         text=Datafile.read()  
        
   
def HTML_ClEAN(text):
        
    soup=BeautifulSoup(text,'html.parser')
    return soup.get_text
        

   # for removing  unnecessary code snippets, ,links, URL...
def remove_CodeSnippet(text):
      
    return re.sub('<pre><code>.*?</code></pre>', '', str(text))

    
    #replacing paragraph and next line headers with a blank string
def remove_Para(text):
      
    text= re.sub('</p>', '', str(text))
    text=  re.sub('\\n', '', str(text))
    text=  re.sub('<p>', '', str(text))
    return text
      
  
#implementing the De-noise Functions to clean the SampleData
def De_noise(text):
    text= HTML_ClEAN(text)
    text= remove_CodeSnippet(text)
    text= remove_Para(text)
    return text
         

#Non-Ascii Words are ignored for better accuracy purpose        
def is_Non_Ascii(ProcessedSampleBodyData):
    NewProcessedSampleBodyData = []
    for word in ProcessedSampleBodyData:
       temp = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
       NewProcessedSampleBodyData.append(temp)
    return NewProcessedSampleBodyData


#converting everyword to lowercase to remove redundancy for ex-is & IS
def Case_lower(ProcessedSampleBodyData):
    NewProcessedSampleBodyData = []
    for word in ProcessedSampleBodyData:
        Temp = word.lower()
        NewProcessedSampleBodyData.append(Temp)
    return NewProcessedSampleBodyData


#removing Punctuation like,0-;] for better data quality
def TextClean(ProcessedSampleBodyData):
    NewProcessedSampleBodyData = []
    for word in ProcessedSampleBodyData:
        temp = re.sub(r'[^\w\s]', '', str(word))
        if  NewProcessedSampleBodyData != '':
             NewProcessedSampleBodyData .append(temp)
    return  NewProcessedSampleBodyData 


#removing Numbers for better tag prediction
def Number_Removal(ProcessedSampleBodyData):
    use = inflect.engine()
    NewProcessedSampleBodyData = []
    for word in ProcessedSampleBodyData:
        if word.isdigit():
          temp  = use.number_to_words(word)
          NewProcessedSampleBodyData.append(temp)
        else:
            NewProcessedSampleBodyData.append(word)
    return NewProcessedSampleBodyData

nltk.download('stopwords')

#filtering out StopWords to before processing natural data
def StopWord_Removal(ProcessedSampleBodyData):
    
     NewProcessedSampleBodyData = []
     for word in ProcessedSampleBodyData:
        if word not in stopwords.words('english'):
           NewProcessedSampleBodyData.append(word)
     return  NewProcessedSampleBodyData


def WordProcessing(Body_word):

    Body_word=is_Non_Ascii(Body_word)
    Body_word=Case_lower(Body_word)
    Body_word=TextClean(Body_word)
    Body_word=Number_Removal(Body_word)
    Body_word=StopWord_Removal(Body_word)

    return Body_word
  
    detext= De_noise(text)   
    print(detext)
    newtext= WordProcessing(detext) 
    print(newtext)
    
    nltk.download('punkt')
    
    #Tokenising the sampledata
    #Tokenising is converting text to words
    ProcessedSampleBodyData = nltk.word_tokenize(text)
    print( ProcessedSampleBodyData)
 
    #BodyWordCount=Counter(ProcessedSampleBodyData)
       
    ProcessedBodyWord= WordProcessing(ProcessedSampleBodyData)

    count=Counter( ProcessedBodyWord)
    print(count)
