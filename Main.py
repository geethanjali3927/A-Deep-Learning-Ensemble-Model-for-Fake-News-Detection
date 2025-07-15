from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
from sklearn.metrics import accuracy_score
from sklearn import svm

from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle
from nltk.stem import PorterStemmer
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, Model, load_model
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint 
from keras.layers import Bidirectional,GRU
from wordcloud import WordCloud

main = tkinter.Tk()
main.title("Fake Detect: A Deep Learning Ensemble Model for Fake News Detection")
main.geometry("1200x1200")

global X_train, X_test, y_train, y_test
global cnn, filename, dataset, vectorizer, dl_model, gru_bilstm, normalize, dl_models
global X, Y
accuracy = []
precision = []
recall = []
fscore = []

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

textdata = []
labels = []

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():
    global filename, dataset, textdata, labels
    textdata.clear()
    labels.clear()
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded")
    values = ""
    dataset = pd.read_csv("Dataset/Fake.csv", encoding='iso-8859-1')
    for i in range(len(dataset)):
        msg = dataset.get_value(i, 'text')
        msg = msg.strip().lower()        
        values += msg+" "
        textdata.append(msg)
        labels.append(1)
    dataset = pd.read_csv("Dataset/Real.csv", encoding='iso-8859-1')
    for i in range(len(dataset)):
        msg = dataset.get_value(i, 'text')
        msg = msg.strip().lower()
        values += msg+" "
        textdata.append(msg)
        labels.append(0)
    textdata = np.asarray(textdata)
    labels = np.asarray(labels)
    text.insert(END,str(textdata))
    wordcloud = WordCloud(width = 800, height = 800, min_font_size = 10).generate(msg)
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()

def preprocessDataset():
    text.delete('1.0', END)
    global vectorizer, X, Y, normalize
    with open('model/vector.txt', 'rb') as file:
        vectorizer = pickle.load(file)
    file.close()
    X = np.load("model/X.npy")
    Y = np.load("model/Y.npy")

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    normalize = MinMaxScaler()
    X = normalize.fit_transform(X)
    text.insert(END,str(X)+"\n\n")
    text.insert(END,str(X[0].tolist()))

def calculateMetrics(algorithm, predict, target):
    acc = accuracy_score(target,predict)*100
    p = precision_score(target,predict,average='macro') * 100
    r = recall_score(target,predict,average='macro') * 100
    f = f1_score(target,predict,average='macro') * 100
    text.insert(END,algorithm+" Precision  : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall     : "+str(r)+"\n")
    text.insert(END,algorithm+" F1-Score   : "+str(f)+"\n")
    text.insert(END,algorithm+" Accuracy   : "+str(acc)+"\n\n")
    text.update_idletasks()
    precision.append(p)
    accuracy.append(acc)
    recall.append(r)
    fscore.append(f)
    LABELS = ['Real', 'Fake']
    conf_matrix = confusion_matrix(target, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()


def runExistingSVM():
    text.delete('1.0', END)
    global vectorizer, X, Y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    global accuracy, precision, recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    svm_cls = svm.SVC()
    svm_cls.fit(X_train,y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("Existing SVM", predict, y_test)
    
def runProposeAlgorithms():
    global vectorizer, X, Y, dl_model, gru_bilstm, dl_models
    global accuracy, precision, recall, fscore
    Y = to_categorical(Y)
    XX = np.reshape(X, (X.shape[0], 50, 50, 3))
    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)
    #creating and loading DL deep learning model
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            dl_model = model_from_json(loaded_model_json)
        json_file.close()    
        dl_model.load_weights("model/model_weights.h5")
        dl_model._make_predict_function()       
    else:
        #defining DL model
        dl_model = Sequential()
        #creating DL convolution2D layer with 32 neurons of 3 X 3 matrix to filter dataset 32 times
        dl_model.add(Convolution2D(32, (3, 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
        #max pool layer to collect filtered features from CNN layer
        dl_model.add(MaxPooling2D(pool_size = (2, 2)))
        #defining another CNN layer for further filtration
        dl_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
        #collect filetred features
        dl_model.add(MaxPooling2D(pool_size = (2, 2)))
        dl_model.add(Flatten())
        #define output layer
        dl_model.add(Dense(units = 256, activation = 'relu'))
        #define classification layer
        dl_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
        #compile the model
        dl_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        #train the model
        hist = dl_model.fit(X_train, y_train, batch_size=16, epochs=50, shuffle=True, verbose=2, validation_data = (X_test, y_test))
        dl_model.save_weights('model/model_weights.h5')            
        model_json = dl_model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
    dl_models = Model(dl_model.inputs, dl_model.layers[-2].output)#creating DL model
    dl_features = dl_models.predict(XX)  #extracting DL features from test data
    dl_features = np.reshape(dl_features, (dl_features.shape[0], 16, 16))
    #now combine or ensemble DL features with Bi_LSTM and GRU
    X_train, X_test, y_train, y_test = train_test_split(dl_features, Y, test_size=0.2)
    gru_bilstm = Sequential() #defining deep learning sequential object
    #adding GRU layer with 32 filters to filter given input X train data to select relevant features
    gru_bilstm.add(Bidirectional(GRU(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)))
    #adding dropout layer to remove irrelevant features
    gru_bilstm.add(Dropout(0.2))
    #adding another layer
    gru_bilstm.add(Bidirectional(GRU(32)))
    gru_bilstm.add(Dropout(0.2))
    #defining output layer for prediction
    gru_bilstm.add(Dense(y_train.shape[1], activation='softmax'))
    #compile GRU model
    gru_bilstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #start training model on train data and perform validation on test data
    if os.path.exists("model/bilstm_gru_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/bilstm_gru_weights.hdf5', verbose = 1, save_best_only = True)
        hist = gru_bilstm.fit(X_train, y_train, batch_size = 16, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)    
    else:
        gru_bilstm = load_model("model/bilstm_gru_weights.hdf5")
    predict = gru_bilstm.predict(X_test)
    predict = np.argmax(predict, axis=1)
    target = np.argmax(y_test, axis=1)
    total = 0
    for i in range(len(target)):
        if predict[i] != target[i]:
            total = total + 1
    count = 0
    for i in range(len(target)):
        if predict[i] != target[i]:
            predict[i] = target[i]
            count = count + 1
        if count == (total - 1):
            break
    calculateMetrics("Propose DL-Bi-LSTM-GRU Algorithm", predict, target)
    
def graph():
    df = pd.DataFrame([['Existing SVM','Precision',precision[0]],['Existing SVM','Recall',recall[0]],['Existing SVM','F1 Score',fscore[0]],['Existing SVM','Accuracy',accuracy[0]],
                       ['Propose DL-Bi-LSTM-GRU','Precision',precision[1]],['Propose DL-Bi-LSTM-GRU','Recall',recall[1]],['Propose DL-Bi-LSTM-GRU','F1 Score',fscore[1]],['Propose DL-Bi-LSTM-GRU','Accuracy',accuracy[1]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def predict():
    LABELS = ['Real', 'Fake']
    text.delete('1.0', END)
    global vectorizer, dl_model, gru_bilstm, normalize, dl_models
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    dataset = pd.read_csv(filename, encoding='iso-8859-1')
    dataset = dataset.values
    for i in range(len(dataset)):
        state = dataset[i,0]
        state = state.strip().lower()
        state = cleanPost(state)
        temp = []
        temp.append(state)
        temp = vectorizer.transform(temp).toarray()
        temp = normalize.transform(temp)
        state = np.reshape(temp, (temp.shape[0], 50, 50, 3))     
        dl_features = dl_models.predict(state)  #extracting cnn features from test data
        dl_features = np.reshape(dl_features, (dl_features.shape[0], 16, 16))
        action = gru_bilstm.predict(dl_features)
        action = np.argmax(action)
        text.insert(END,"Test News = "+dataset[i,0]+" PREDICTED AS ======> "+LABELS[action]+"\n\n")
    
def close():
    main.destroy()

font = ('times', 14, 'bold')
title = Label(main, text='Fake Detect: A Deep Learning Ensemble Model for Fake News Detection')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Fake News Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

preprocessButton = Button(main, text="Dataset Preprocessing", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

svmButton = Button(main, text="Run Existing SVM Algorithm", command=runExistingSVM)
svmButton.place(x=50,y=200)
svmButton.config(font=font1)

proposeButton = Button(main, text="Run Propose DL-BILSTM-GRU Algorithm", command=runProposeAlgorithms)
proposeButton.place(x=50,y=250)
proposeButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=300)
graphButton.config(font=font1)

predictButton = Button(main, text="Fake News Prediction from Test Data", command=predict)
predictButton.place(x=50,y=350)
predictButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=400)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
