#---------------- importing data and creating a dataframe--------------------
import pandas as pd
fake_df  = pd.read_csv('database/Fake.csv')
real_df = pd.read_csv('database/Real.csv')
# print(fake_df.info())
# print(real_df.info())



#---------------modifying dataframes according to our needs------------------------
fake_df['label'] = ["fake" for i in range(fake_df.shape[0]) ]
fake_df.drop(columns=['subject','date'],inplace=True,axis=1)
# print(fake_df.head())

real_df['label'] = ["real" for i in range(real_df.shape[0]) ]
real_df.drop(columns=['subject','date'],inplace=True,axis=1)
# print(real_df.head())



#--------------------making a parent dataframe-----------------
df = pd.concat([real_df,fake_df],axis=0)
df['b_label']=df['label'].map({'real':1,'fake':0})
# print(df.info())
# print(df.head())
# print(df.tail())


#--------------------------------splitting the data --------------------------------
from sklearn.model_selection import train_test_split
x=df[['text','title']]      #[[]] is used bcs a list of columns is passed
y=df['b_label']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=47)
print(x_train.shape)
print(y_train.shape)


#-------------------------------- importing models--------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

#--------------------------------pipeline--------------------------------
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
#************** ColumnTransformer is a utility in scikit-learn that allows you to apply different preprocessing techniques to different columns of your dataset.***********


# Define the column-wise vectorization
preprocessor = ColumnTransformer(
    transformers=[
        ('title_vec', TfidfVectorizer(), 'title'),
        ('text_vec', TfidfVectorizer(), 'text'),
    ]
)
steps=[ ('vectorsier',preprocessor),
       ('model',LogisticRegression()) ]
pipe=Pipeline(steps=steps)

# pipe.fit(x_train,y_train)
# print(pipe.get_params())

# ----------------upgradations with Cross validation ----------------
#************GridSearchCV expects each key in param_grid to map to a list of candidate values that it will iterate over.***********
parameters=[
            {'model':[LogisticRegression()],
             'model__C': [0.01, 0.1, 1, 10],
            'model__penalty': ['l2'],
            'model__solver': ['liblinear', 'saga'],
            'model__class_weight': ['balanced', None]
            } ,
            
            {'model':[RandomForestClassifier()],
             'model__criterion':['gini','entropy','log_loss'],
             'model__max_depth':[i for i in range(1,10)]
             } ,

            {'model':[MultinomialNB()],
              'model__alpha': [0.001, 0.01, 0.1, 1, 10],
                'model__fit_prior': [True, False]
            }

             ]
# ----------------upgradations with Cross validation ----------------
scoring = {'accuracy': 'accuracy', 'f1': 'f1'}

cv_obj = GridSearchCV(
    estimator=pipe,
    param_grid=parameters,
    cv=4,
    scoring=scoring,
    refit='accuracy',
    n_jobs=6,  #tells how much cores to use
    return_train_score=True
)

cv_obj.fit(x_train, y_train)

# ---------------- Best model ----------------
print(f"\n✅ Best Parameters: {cv_obj.best_params_}")
print(f"✅ Best Accuracy: {cv_obj.best_score_:.4f}")

# ---------------- Summarizing results ----------------
results = pd.DataFrame(cv_obj.cv_results_)

# Select only useful columns
results_summary = results[
    [
        "param_model",
        "mean_test_accuracy",
        "mean_test_f1",
        "rank_test_accuracy",
        "rank_test_f1"
    ]
].sort_values("rank_test_accuracy")

print("\n📊 Performance of all models:\n")
print(results_summary.head(10))  # Show top 10 configs

# ---------------- Evaluate best model ----------------
best_model = cv_obj.best_estimator_
y_pred = best_model.predict(x_test)

from sklearn.metrics import accuracy_score, f1_score

print("\n🚀 Final Evaluation on Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")


# ----------------visualising the comparisions---------------------------
import matplotlib.pyplot as plt
import numpy as np

# create readable model names and limit to top-N rows
plot_df = results_summary.copy()
plot_df['model_name'] = plot_df['param_model'].astype(str).str.replace(r'\(.*', '', regex=True)
topn = min(10, len(plot_df))
plot_df = plot_df.head(topn).reset_index(drop=True)

# grouped bar plot
x = np.arange(len(plot_df))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, plot_df['mean_test_accuracy'], width, label='mean_test_accuracy', color='tab:blue')
ax.bar(x + width/2, plot_df['mean_test_f1'], width, label='mean_test_f1', color='tab:orange')

ax.set_xticks(x)
ax.set_xticklabels(plot_df['model_name'], rotation=45, ha='right')
ax.set_ylabel('Score')
ax.set_title('Model performance (mean test scores)')
ax.legend()
plt.tight_layout()
plt.savefig(r'C:\Users\LENOVO\Pictures\model_perf.jpg', dpi=300, bbox_inches='tight')
plt.close(fig)


#--------------------------- making the model ready--------------------------------
import pickle
with open('news_auth_model.pkl','wb') as file:
  pickle.dump(model,file)
