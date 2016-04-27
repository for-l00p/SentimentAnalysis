__author__ = 'Shubham'
import numpy as np
import csv
import twittersegment
import nltk
import spellchecker
a = np.zeros((2000, 8))
b = np.zeros((2000,1))
POSITIVE = ["*O", "*-*", "*O*", "*o*", "* *",
                ":P", ":D", ":d", ":p",
                ";P", ";D", ";d", ";p",
                ":-)", ";-)", ":=)", ";=)",
                ":<)", ":>)", ";>)", ";=)",
                "=}", ":)", "(:;)",
                "(;", ":}", "{:", ";}",
                "{;:]",
                "[;", ":')", ";')", ":-3",
                "{;", ":]",
                ";-3", ":-x", ";-x", ":-X",
                ";-X", ":-}", ";-=}", ":-]",
                ";-]", ":-.)",
                "^_^", "^-^"]
NEGATIVE = [":(", ";(", ":'(",
                "=(", "={", "):", ");",
                ")':", ")';", ")=", "}=",
                ";-{{", ";-{", ":-{{", ":-{",
                ":-(", ";-(",
                ":,)", ":'{",
                "[:", ";]"
                ]
import re, collections

def train1(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model
def words(text): return re.findall('[a-z]+', text.lower())


negative = []
positive = []
negation = []
train = []
test=[]
atest=np.zeros((100,8))
btest=np.zeros((100, 1))
special=["RB","RBR","RBS","JJ","JJR","JJS","VB","VBD","VBG","VBN","VBP","VBZ"]


def getnegative():
    global negative
    fp = open("data/negative.txt")
    x = fp.read()
    negative = x.split('\n')


def getnegation():
    global negation
    fp = open("data/negation.txt")
    x = fp.read()
    negation = x.split('\n')


def getpositive():
    global positive
    fp = open("data/positive.txt")
    x = fp.read()
    positive = x.split('\n')


def trim():
    j = 0
    global train

    while j < len(train):
        train[j] = train[j].strip()
        k=0
        while k<len(POSITIVE):
            if POSITIVE[k] in train[j]:
                a[j][7]+=1
            k+=1
        k=0
        while k<len(NEGATIVE):
            if NEGATIVE[k] in train[j]:
                a[j][7]-=1
            k+=1
        x = train[j].replace('.', ' ').replace('?', ' ').replace(',', ' ').replace(')', ' ').replace(':', ' ').replace(
            ';', ' ').replace('-', ' ').replace('!', ' ').replace('\t', ' ').replace('*',' ')

        newstring = ""
        y = " ".join(x.split())
        # print x.split
        y = y.split()
        k = 0
        while k < len(y):
            if '@' not in y[k] and '&' not in y[k] and 'http' not in y[k]:
                newstring += y[k] + " "
            k += 1

        newstring = newstring.strip()

        train[j] = newstring
        # print newstring
        j += 1
def trimtestdata():
    j = 0
    global test

    while j < len(test):
        test[j] = test[j].strip()

        k=0
        while k<len(POSITIVE):
            if POSITIVE[k] in test[j]:
                atest[j][7]+=1
            k+=1
        k=0
        while k<len(NEGATIVE):
            if NEGATIVE[k] in test[j]:
                atest[j][7]-=1
            k+=1


        x = test[j].replace('.', ' ').replace('?', ' ').replace(',', ' ').replace(')', ' ').replace(':', ' ').replace(
            ';', ' ').replace('-', ' ').replace('!', ' ').replace('\t', ' ').replace('*',' ')

        newstring = ""
        y = " ".join(x.split())
        # print x.split
        y = y.split()
        k = 0
        while k < len(y):
            if '@' not in y[k] and '&' not in y[k] and 'http' not in y[k]:
                newstring += y[k] + " "
            k += 1

        newstring = newstring.strip()

        test[j] = newstring
        # print newstring
        j += 1


def countpositivenegtive():
    i = 0
    while i < len(train):
        str = train[i].split(' ')

        j = 0
        text = nltk.word_tokenize(train[i])
        tk = nltk.pos_tag(text)

        sp1=0
        sp2=0
        print i

        while j < len(str):
            str[j]=spellchecker.correct(str[j],model)
            #print str[j]
            if str[j] in positive:
                if tk[j][1] in special:
                    sp1+=1

                a[i][0] += 1
            if str[j] in negative:
                if tk[j][1] in special:
                    sp2+=1
                a[i][1] += 1
            if str[j] in negation:
                # print str[j]
                a[i][2] += 1


            if '#' in str[j]:
                #print str[j]
                x = twittersegment.tweet(str[j][1:])
                k = 0
                while k < len(x):
                    if x[k] == '':
                        k += 1
                        continue
                    else:
                        if x[k] in positive:
                            a[i][3] += 1
                        if x[k] in negative:
                            a[i][4] += 1

                    k += 1
                    # print x
                    # print a[i][3],a[i][4]

            j += 1
        if sp1>0 and sp2>0:
            a[i][5]=0
            a[i][6]=0
        else:
            if sp1>0:
                a[i][5]=1
                a[i][6]=1
            else:
                if sp2>0:
                    a[i][5]=-1
                    a[i][6]=1
                else:

                    a[i][5]=0
                    a[i][6]=0

        # print train[i]
        # print a[i][0],
        # print a[i][1],
        # print a[i][2], a[i][3], a[i][4],a[i][5],a[i][6]
        i += 1
def countpositivenegtivetestdata():
    global test
    i = 0
    while i < len(test):
        str = test[i].split(' ')

        j = 0
        text = nltk.word_tokenize(test[i])
        tk = nltk.pos_tag(text)

        sp1=0
        sp2=0

        while j < len(str):
            str[j]=spellchecker.correct(str[j],model)

            if str[j] in positive:
                if tk[j][1] in special:
                    sp1+=1

                atest[i][0] += 1
            if str[j] in negative:
                if tk[j][1] in special:
                    sp2+=1
                atest[i][1] += 1
            if str[j] in negation:
                # print str[j]
                atest[i][2] += 1


            if '#' in str[j]:
                #print str[j]
                x = twittersegment.tweet(str[j][1:])
                k = 0
                while k < len(x):
                    if x[k] == '':
                        k += 1
                        continue
                    else:
                        if x[k] in positive:
                            atest[i][3] += 1
                        if x[k] in negative:
                            atest[i][4] += 1

                    k += 1
                    # print x
                    # print a[i][3],a[i][4]

            j += 1
        if sp1>0 and sp2>0:
            atest[i][5]=0
            atest[i][6]=0
        else:
            if sp1>0:
                atest[i][5]=1
                atest[i][6]=1
            else:
                if sp2>0:
                    atest[i][5]=-1
                    atest[i][6]=1
                else:

                    atest[i][5]=0
                    atest[i][6]=0

        # print train[i]
        # print atest[i][0],
        # print atest[i][1],
        # print atest[i][2], atest[i][3], atest[i][4],atest[i][5],atest[i][6]
        i += 1

def readdata():
    global train,b
    fp = open("data/train.csv")
    flag = 0
    count1=0
    for row in csv.reader(fp):
        if (flag == 0):
            flag = 1
            continue
        b[count1][0]=row[1]
        train.append(row[2])
        count1+=1

    b=b.flatten()
    print b
    trim()
    countpositivenegtive()

def readtestdata():
    global test,btest
    fp = open("data/test.csv")
    flag = 0
    count1=0
    for row in csv.reader(fp):


        test.append(row[2])
        btest[count1][0]=row[1]
        count1+=1


    btest=btest.flatten()
    #print b
    trimtestdata()
    countpositivenegtivetestdata()

    # print train
def performbayes():
    global a,b,atest
    from sklearn import naive_bayes
    from sklearn import ensemble
    from sklearn import svm

    #model=naive_bayes.GaussianNB()
    model=ensemble.RandomForestClassifier()
    #model=svm.SVC()
    import scipy
    '''j=0
    while j<len(b):
        if b[j]==0:
            b[j]=-1
        j+=1
    j=0
    while j<len(btest):
        if btest[j]==0:
            btest[j]=-1
        j+=1'''

    # scipy.io.savemat('data/adata.mat', mdict={'a': a})
    # scipy.io.savemat('data/bdata.mat', mdict={'b': b})
    # scipy.io.savemat('data/atestdata.mat', mdict={'atest': atest})
    # scipy.io.savemat('data/btestdata.mat', mdict={'btest': btest})
    model.fit(a,b)
    ans=model.predict(atest)

    count1=0.0
    i=0
    print ans
    print btest

    while i<len(ans):
        if ans[i]==btest[i]:
            count1+=1
        i+=1

    print count1
    print count1/float(len(ans))




if __name__ == '__main__':
    global model
    model = train1(words(file('data/big.txt').read()))
    getnegative()
    getpositive()
    getnegation()
    # print negation
    readdata()
    readtestdata()
    performbayes()

    # x="hello   kjkj "
    # print x.split()
