# IR Assignment 2
# Umair Amir
# K20-0133

import glob
import nltk
import os
import string
from nltk.stem import PorterStemmer
import re
import math 
Dictionary = {} #Create a global dictionary
DocVectors = {} #Create a global dictionary for Document Vectors


def FileRead(): 
    Folder = 'Dataset'
    Pattern = '*.txt' 
    FList = glob.glob(os.path.join(Folder, Pattern)) #Finding all Files in the given Folder 
    for Path in FList: 
        with open(Path, 'r') as file: 
            FileContents = file.read() #Reading File text
            FileContents = FileContents.lower()
            File_name = Path.strip("Dataset\\.txt")
            FileContents = PunctuationRemove(FileContents)# Removing Punctuations
            FileContents = FileContents.split() # Tokenizing string
            Stemmer = PorterStemmer()
            FileStem = []
            #Applying Stemming to all the tokens
            for words in list(FileContents):
                FileStem.append(Stemmer.stem(words))
            File_name = int(File_name)
            Dictionary = DictionaryBuilder(FileStem,File_name)
            Dictionary = sorted(Dictionary.items()) # Sorting the Dictionary by tokens
            Dictionary = dict(Dictionary)
    # Initializing all Document Vectors with 0 for every word
    for i in range(1,31):
         DocVectors[i] = [0] * len(Dictionary)
    return Dictionary


def PunctuationRemove(File):
    File = File.replace('-', ' ') # Replacing hyphens with spaces
    File = File.translate(str.maketrans("", "", string.punctuation))
    return(File)


def DictionaryBuilder(File,File_Name):
    Stop = open(r'C:\Users\umair\Desktop\C\IR Assignment 1\Stopword-List.txt', 'r')
    StopContents = Stop.read()
    StopContents = StopContents.split()
    for words in File: # Building Dictionary
        if(words not in StopContents):
            if(words not in Dictionary): # First time a word is added to Dictionary
                Dictionary[words] = {}
                Dictionary[words][File_Name] = 1 # Setting Term Frequency for the document to 1
            else:
                if(File_Name not in Dictionary[words]):
                    Dictionary[words][File_Name] = 1 # Setting Term Frequency for the document to 1
                else:
                    Dictionary[words][File_Name] += 1 # Incrementing Term Frequency
    return Dictionary   

def BuildDocumentVectors():
    for Index, Key in enumerate(Dictionary): # Traversing through words in Dictionary
        for DocKeys in DocVectors.keys(): # Traversing through all Documents
            if(DocKeys in Dictionary[Key]):
                DocFreq = len(Dictionary[Key]) 
                InvertedDocFreq = round(math.log(len(DocVectors) / DocFreq, 10),2) # Calculating Inverted Document Frequency
                TfIdf = InvertedDocFreq * Dictionary[Key][DocKeys]
                DocVectors[DocKeys][Index] = TfIdf


def QueryProcessor(Query):
    Query = Query.split()
    Query = QueryStemmer(Query)
    QueryVector = [0] * len(Dictionary) # Initializing Query Vector
    QueryDict = {}
    for words in Query: # Building Dictionary for Query
        if(words not in QueryDict): # First time a word is added to Dictionary
            QueryDict[words] = 1
        else:
            QueryDict[words] += 1
    for Index, Key in enumerate(Dictionary): # Traversing Dictionary
            if(Key in QueryDict):
                DocFreq = len(Dictionary[Key])
                InvertedDocFreq = math.log(len(DocVectors) / DocFreq, 10)
                TfIdf = InvertedDocFreq * QueryDict[Key]
                QueryVector[Index] = TfIdf
    return QueryVector


def QueryStemmer(Query):
    StemQuery = []
    Stop = open(r'C:\Users\umair\Desktop\C\IR Assignment 1\Stopword-List.txt', 'r')
    StopContents = Stop.read()
    StopContents = StopContents.split()
    Stemmer = PorterStemmer()
    Query = [Val for Val in Query if Val not in StopContents]
    for words in Query:
        StemQuery.append(Stemmer.stem(words))
    return StemQuery  

# Calculating Eucilidean Length for a Vector
def EucDist(Vector):
    Sum = 0
    for i in Vector:
        Sum += i ** 2
    return(math.sqrt(Sum))

def Solver(Query):
    ResultList = []
    QueryEucDist = EucDist(Query) # Calculating Eucilidean Length for the Query
    if(QueryEucDist == 0): # Return empty list if none of the words in the Query are in the Dictionary
        return ResultList
    for Doc in DocVectors.keys():
        Cosine = 0
        DotProduct = 0
        DocEucDist = EucDist(DocVectors[Doc]) # Calculating Eucilidean Length for a given Document
        for i in range(0,len(Dictionary)):
            if(Query[i] == 0 or DocVectors[Doc][i] == 0): # Skip calculation if one of the TF-IDFs are 0
                continue
            else:
                DotProduct = DotProduct + (Query[i] * DocVectors[Doc][i])
        Cosine = DotProduct / (QueryEucDist * DocEucDist)
        if(Cosine > 0.05): # Threshold
            ResultList.append((Doc,Cosine))
    ResultList = sorted(ResultList, key=lambda x:-x[1]) # Sort results according to Cosine value
    return ResultList

Dictionary = FileRead()
BuildDocumentVectors()
Query = ''
while(1):
    Query = input("Enter Query(Type -1 to exit): ")
    if(Query == '-1'):
        break
    Query = QueryProcessor(str(Query))
    print(Solver(Query))


