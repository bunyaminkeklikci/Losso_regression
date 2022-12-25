import pandas as pd
from sklearn.datasets import load_boston

df=load_boston()
data=pd.DataFrame(df.data,columns=df.feature_names)

veri=data.copy()

veri["PRICE"]=df.target

y=veri["PRICE"]
X=veri.drop(columns="PRICE",axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import Lasso

lasso_model=Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)

#tahmin=lasso_model.predict(X_test)
#import sklearn.metrics as mt
#print(mt.r2_score(y_test,tahmin)) bu çıktı lasso model score X_test,y_test sonucu ile aynı

print(lasso_model.score(X_train,y_train))
print(lasso_model.score(X_test,y_test))

#Lambda tahmin edebilmek için crossvalidation(çarpraz doğrulama)
#en iyi lambda değerini bulcaz
from sklearn.linear_model import LassoCV

lamb =LassoCV(cv=10,max_iter=10000).fit(X_train,y_train).alpha_
#print(lamb)

lasso_model2=Lasso(alpha=lamb)
lasso_model2.fit(X_train,y_train)

print(lasso_model2.score(X_train,y_train))
print(lasso_model2.score(X_test,y_test))

