#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from collections import Counter
import numpy as np
import pandas as pd


# In[2]:


folder='mails/'
files=os.listdir(folder)
len(files)


# In[3]:


files=os.listdir(folder)
files[:5]


# In[4]:


emails=[folder + file for file in files]
emails[:5]


# In[5]:


words=[]

for email in emails:
    f=open(email,encoding='latin-1')
    blob=f.read()
    words=words+blob.split()
len(words)


# In[6]:


for i in range(len(words)):
    if not words[i].isalpha():
        words[i]=""


# In[7]:


word_dict=Counter(words)
len(word_dict)


# In[8]:


del word_dict[""]


# In[9]:


word_dict=word_dict.most_common(3000)
word_dict[:5]


# In[10]:


features=[]
labels=[]

for email in emails:
    f=open(email,encoding='latin-1')
    blob=f.read().split(" ")
    data=[]
    
    for i in word_dict:
        data.append(blob.count(i[0]))
    features.append(data)
    
    if 'spam'in email:
        labels.append(1)
    if 'ham' in email:
        labels.append(0)

print("Features", len(features))
print("Labels", len(labels))


# In[11]:


# Independent variables

features=np.array(features)
print(features.shape)
features[:5]


# In[12]:


# Dependent variables

labels=np.array(labels)
print(labels.shape)
labels


# In[ ]:





# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(features,labels,test_size=0.2,random_state=9)
X_train.shape ,X_test.shape ,y_train.shape ,y_test.shape


# In[14]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[15]:


lst = [3,4,5,6,7,8]
accuracies = []
for i in lst:   
    knn = KNeighborsClassifier(n_neighbors= i)
    knn.fit(X_train, y_train)
    
    accuracies.append(round(knn.score(X_test, y_test),4))
    
accuracies


# In[16]:


acc = pd.DataFrame({"n_neighbors": lst,
                   "Accuracies" : accuracies})
acc


# In[17]:


knn = KNeighborsClassifier(n_neighbors= 6)
knn.fit(X_train, y_train)


# In[18]:


predict= knn.predict(X_test)
m=confusion_matrix(predict,y_test)
print(m)


# In[19]:


import seaborn as sns
sns.heatmap(m, annot=True);


# In[20]:


from sklearn.metrics import precision_score, recall_score
print("precision_score",precision_score(predict,y_test))
print("recall_score",recall_score(predict,y_test))


# In[21]:


new_mail = """Subject: re : entex transistion
thanks so much for the memo . i would like to reiterate my support on two key
issues :
1 ) . thu - best of luck on this new assignment . howard has worked hard and
done a great job ! please don ' t be shy on asking questions . entex is
critical to the texas business , and it is critical to our team that we are
timely and accurate .
2 ) . rita : thanks for setting up the account team . communication is
critical to our success , and i encourage you all to keep each other informed
at all times . the p & l impact to our business can be significant .
additionally , this is high profile , so we want to assure top quality .
thanks to all of you for all of your efforts . let me know if there is
anything i can do to help provide any additional support .
rita wynne
12 / 14 / 99 02 : 38 : 45 pm
to : janet h wallis / hou / ect @ ect , ami chokshi / corp / enron @ enron , howard b
camp / hou / ect @ ect , thu nguyen / hou / ect @ ect , kyle r lilly / hou / ect @ ect , stacey
neuweiler / hou / ect @ ect , george grant / hou / ect @ ect , julie meyers / hou / ect @ ect
cc : daren j farmer / hou / ect @ ect , kathryn cordes / hou / ect @ ect , rita
wynne / hou / ect , lisa csikos / hou / ect @ ect , brenda f herod / hou / ect @ ect , pamela
chambers / corp / enron @ enron
subject : entex transistion
the purpose of the email is to recap the kickoff meeting held on yesterday
with members from commercial and volume managment concernig the entex account :
effective january 2000 , thu nguyen ( x 37159 ) in the volume managment group ,
will take over the responsibility of allocating the entex contracts . howard
and thu began some training this month and will continue to transition the
account over the next few months . entex will be thu ' s primary account
especially during these first few months as she learns the allocations
process and the contracts .
howard will continue with his lead responsibilites within the group and be
available for questions or as a backup , if necessary ( thanks howard for all
your hard work on the account this year ! ) .
in the initial phases of this transistion , i would like to organize an entex
" account " team . the team ( members from front office to back office ) would
meet at some point in the month to discuss any issues relating to the
scheduling , allocations , settlements , contracts , deals , etc . this hopefully
will give each of you a chance to not only identify and resolve issues before
the finalization process , but to learn from each other relative to your
respective areas and allow the newcomers to get up to speed on the account as
well . i would encourage everyone to attend these meetings initially as i
believe this is a critical part to the success of the entex account .
i will have my assistant to coordinate the initial meeting for early 1 / 2000 .
if anyone has any questions or concerns , please feel free to call me or stop
by . thanks in advance for everyone ' s cooperation . . . . . . . . . . .
julie - please add thu to the confirmations distributions list """


# In[22]:


sample = []
for i in word_dict:
    sample.append(new_mail.split(" ").count(i[0]))


# In[23]:


sample = np.array(sample)


# In[24]:


knn.predict(sample.reshape(1,3000))


# In[ ]:


import joblib
joblib.dump(word_dict, "word_dict.pkl")


joblib.dump(knn, "model.pkl")

