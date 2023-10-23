# data_cleaning
Basic step of Data Preprocessing


    import pandas as pd
    import numpy as np
    
    data=pd.read_csv("Placement_Data.csv")
    data1=data.copy()
    
    x=data1.iloc[:,:-2].values
    y=data1.iloc[:,-1].values
    print(x)
    print(y)
    
    from sklearn.impute import SimpleImputer
    imputer_y=SimpleImputer(missing_values=np.nan,strategy='mean')
    imputer_y.fit(y.reshape(-1,1))
    y=imputer_y.transform(y.reshape(-1,1))
    print(y)
    
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[2])],remainder='passthrough')
    x=np.array(ct.fit_transform(x))
    print(x)
    
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    x[:,6]=le.fit_transform(x[:,6])
    print(x)
    
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    x=sc.fit_transform(x)
    print(x)
    
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    print(x_train)
