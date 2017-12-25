
# coding: utf-8

# # Project 3 - Classification
# Welcome to the third project of Data 8!  You will build a classifier that guesses whether a movie is romance or action, using only the numbers of times words appear in the movies's screenplay.  By the end of the project, you should know how to:
# 
# 1. Build a k-nearest-neighbors classifier.
# 2. Test a classifier on data.
# 
# ### Logistics
# 
# 
# **Deadline.** This project is due at 11:59pm on Thursday 11/30. You can earn an early submission bonus point by submitting your completed project by Wednesday 11/29. It's **much** better to be early than late, so start working now.


# Run this cell to set up the notebook, but please don't change it.

import numpy as np
import math
from datascience import *

# These lines set up the plotting functionality and formatting.
import matplotlib
matplotlib.use('Agg', warn=False)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# These lines load the tests.
from client.api.notebook import Notebook
ok = Notebook('project3.ok')
_ = ok.auth(inline=True)


# # 1. The Dataset
# 
# In this project, we are exploring movie screenplays. We'll be trying to predict each movie's genre from the text of its screenplay. In particular, we have compiled a list of 5,000 words that might occur in the dialog of a movie. For each movie, our dataset tells us the frequency with which each of these words occurs in its screenplay. All words have been converted to lowercase.
# 
# Run the cell below to read the `movies` table. **It may take up to a minute to load.**

# In[2]:


movies = Table.read_table('movies.csv')
movies.where("Title", "the matrix").select(0, 1, 2, 3, 4, 5, 10, 30, 5005)


title_index = movies.index_by('Title')
def row_for_title(title):
    """Return the row for a title, similar to the following expression (but faster)
    
    movies.where('Title', title).row(0)
    """
    return title_index.get(title)[0]


# For example, the fastest way to find the frequency of "hey" in the movie *The Terminator* is to access the `'hey'` item from its row. Check the original table to see if this worked for you!

# In[4]:

row_for_title('the terminator').item('hey') 

# In[9]:


# Histogram of the sum of word frequency proportions for each movie.
Table().with_column('Sum', movies.drop(0, 1, 2, 3, 4, 5).apply(sum)).hist()


# This dataset was extracted from [a Kaggle dataset from Cornell University](https://www.kaggle.com/Cornell-University/movie-dialog-corpus). After transforming the dataset (e.g., converting the words to lowercase, removing the naughty words, and converting the counts to frequencies), we created this new dataset containing the frequency of 5000 common words in each movie.

# In[10]:

print('Words with frequencies:', movies.drop(np.arange(6)).num_columns) 
print('Movies with genres:', movies.num_rows)

# In[11]:

vocab_mapping = Table.read_table('stem.csv')
stemmed = np.take(movies.labels, np.arange(3, len(movies.labels)))
vocab_table = Table().with_column('Stem', stemmed).join('Stem', vocab_mapping)
v = vocab_table.take(np.arange(1100, 1110))

vocab_table


# Assigned `percent_unchanged` to the **percentage** of words in `vocab_table` that are the same as their stemmed form (such as "blame" above).

# In[13]:

vocab_table.where('Stem', are.equal_to(vocab_table.column('Word'))).show(200)


# In[20]:

num_unchanged = vocab_table.where('Word', are.equal_to(vocab_table.column('Stem'))).sort("Word",distinct=True).num_rows
percent_unchanged = (num_unchanged/vocab_table.num_rows)*100
print(round(percent_unchanged, 2), '% are unchanged')


# Assigned `stemmed_message` to the stemmed version of the word "alternating".

# In[22]:

# Set stemmed_message to the stemmed version of "alternating" 
stemmed_message = vocab_table.where('Word', are.equal_to('alternating')).column(0).item(0)
stemmed_message


# Assigned `unstemmed_run` to an array of words in `vocab_table` that have "run" as its stemmed form. 

# In[52]:
versions of "run" (which
# should be an array of string).
unstemmed_run = vocab_table.where('Stem', are.equal_to('run')).column(1)
unstemmed_run


# In[54]:

v = vocab_table.where("Stem", are.between("n","o"))
v.with_column("num", v.apply(len, "Word") - v.apply(len, "Stem")).sort("num", descending=True).take(0)


# In[55]:

shortened = vocab_table.with_column("num", vocab_table.apply(len, "Word") - vocab_table.apply(len, "Stem")).sort("num", descending=True)
most_shortened = shortened.where("Stem", are.between("n","o")).column("Word").item(0)
most_shortened


# In[56]:

shortened = vocab_table.with_column("num", vocab_table.apply(len, "Word") - vocab_table.apply(len, "Stem")).sort("num", descending=True)
most_shortened = shortened.where("Stem", are.between("n","o")).column("Word").item(0)

# This will display your answer and its shortened form.
vocab_table.where('Word', most_shortened)


#Splitting the Dataset
# In[58]:
training_proportion = 17/20

num_movies = movies.num_rows
num_train = int(num_movies * training_proportion)
num_valid = num_movies - num_train

train_movies = movies.take(np.arange(num_train))
test_movies = movies.take(np.arange(num_train, num_movies))

print("Training: ",   train_movies.num_rows, ";",
      "Test: ",       test_movies.num_rows)

# Drew a horizontal bar chart with two bars that show the proportion of Romance movies in each dataset.
# In[59]:

def romance_proportion(table):
    """Returns the proportion of movies in a table that have the Romance genre."""
    num_romance = table.where("Genre", are.equal_to("romance")).num_rows
    return num_romance/table.num_rows
 
romance_table = Table().with_columns(
    "Data set", make_array("Training", "Test"),
    "Number", make_array(romance_proportion(train_movies), romance_proportion(test_movies)))
romance_table.barh("Data set", "Number")


train_movies


# Classifying a movie
#
def plot_with_two_features(test_movie, training_movies, x_feature, y_feature):
    test_row = row_for_title(test_movie)
    distances = Table().with_columns(
            x_feature, [test_row.item(x_feature)],
            y_feature, [test_row.item(y_feature)],
            'Color',   ['unknown'],
            'Title',   [test_movie]
        )
    for movie in training_movies:
        row = row_for_title(movie)
        distances.append([row.item(x_feature), row.item(y_feature), row.item('Genre'), movie])
    distances.scatter(x_feature, y_feature, colors='Color', labels='Title', s=30)
    
training = ["titanic", "the avengers"] 
plot_with_two_features("batman returns", training, "money", "feel")
plots.axis([-0.001, 0.0015, -0.001, 0.006])


# In[38]:


batman = row_for_title("batman returns") 
avengers = row_for_title("the avengers") 
romance_distance = np.sqrt((batman.item("money") - avengers.item("money"))**2 + (batman.item("feel") - avengers.item("feel"))**2)
romance_distance

# In[64]:


training = ["the avengers", "titanic", "the terminator"] 
plot_with_two_features("batman returns", training, "money", "feel") 
plots.axis([-0.001, 0.0015, -0.001, 0.006])

# In[65]:


def distance_two_features(title0, title1, x_feature, y_feature):
    row0 = row_for_title(title0)
    row1 = row_for_title(title1)
    return np.sqrt((row0.item(x_feature) - row1.item(x_feature))**2 + (row0.item(y_feature) - row1.item(y_feature))**2)

for movie in make_array("the terminator", "the avengers"):
    movie_distance = distance_two_features(movie, "batman returns", "money", "feel")
    print(movie, 'distance:\t', movie_distance)

# In[ ]:


def distance_from_batman_returns(title):
    """The distance between the given movie and "batman returns", based on the features "money" and "feel".
    
    This function takes a single argument:
      title: A string, the name of a movie.
    """
    
    return distance_two_features("batman returns",title,"money","feel")


x = train_movies.with_column("distance",train_movies.apply(distance_from_batman_returns,"Title"))
x.sort("distance").take(np.arange(0,6))

# In[47]:

x = train_movies.with_column("distance",train_movies.apply(distance_from_batman_returns,"Title"))
x = x.sort("distance").take(np.arange(0,7))
close_movies = Table().with_columns("Title",x.column("Title"),"Genre",x.column("Genre"),"money",x.column("money"),"feel",x.column("feel"))
close_movies


# In[49]:


def most_common(label, table):
    """The most common element in a column of a table.
    
    This function takes two arguments:
      label: The label of a column, a string.
      table: A table.
     
    It returns the most common value in that column of that table.
    In case of a tie, it returns any one of the most common values
    """
    return table.group(label).sort("count",descending=True).column(label).item(0)

most_common('Genre', close_movies)


# In[50]:

close_movies.group("Genre").sort("count",descending=True)

# In[51]:

# # 3. Features
clidean distance still makes sense with more than two features. For `n` different features, we compute the difference between corresponding feature values for two movies, square each of the `n`  differences, sum up the resulting numbers, and take the square root of the sum.

# #### Question 3.1
# Wrote a function to compute the Euclidean distance between two **arrays** of features of *arbitrary* (but equal) length.  Use it to compute the distance between the first movie in the training set and the first movie in the test set, *using all of the features*.  (Remember that the first six columns of your tables are not features.)
# In[47]:


first_train_movie = np.array(train_movies.row(0)).take(np.arange(6,train_movies.num_columns))
first_test_movie = np.array(test_movies.row(0)).take(np.arange(6,test_movies.num_columns))

np.array(test_movies.row(0)[6:])


# In[48]:


def distance(features1, features2):
    """The Euclidean distance between two arrays of feature values."""
    return np.sqrt(sum((features1 - features2)**2))
first_train_movie = np.array(train_movies.row(0)[6:])
first_test_movie = np.array(test_movies.row(0)[6:])
distance_first_to_first = distance(first_train_movie,first_test_movie)
distance_first_to_first



# In[58]:

movie_genre_guess = 1

# In[129]:


# Set my_20_features to an array of 20 features (strings that are column labels)

my_20_features = make_array("write","merri","mom","parti","wed",
                            "huh","nice","hous","madam","darl","captain",
                            "power","secur","system","ship","run","weve","world","cop","command")

train_20 = train_movies.select(my_20_features)
test_20 = test_movies.select(my_20_features)

train_20

# In[118]:

print("Movie:")
test_movies.take(0).select('Title', 'Genre').show()
print("Features:")
test_20.take(0).show()


# In[119]:

# Run this cell to define fast_distances.

def fast_distances(test_row, train_rows):
    assert train_rows.num_columns < 50, "Make sure you're not using all the features of the movies table."
    counts_matrix = np.asmatrix(train_rows.columns).transpose()
    diff = np.tile(np.array(test_row), [counts_matrix.shape[0], 1]) - counts_matrix
    np.random.seed(0) # For tie breaking purposes
    distances = np.squeeze(np.asarray(np.sqrt(np.square(diff).sum(1))))
    eps = np.random.uniform(size=distances.shape)*1e-10 #Noise for tie break
    distances = distances + eps
    return distances




# Ensured that `genre_and_distances` is **sorted in increasing order by distance to the first test movie**.

# In[120]:

train_movies.with_column("Distance",fast_distances(test_20.row(0),train_20))
not_sorted = train_movies.with_column("Distance",fast_distances(test_20.row(0),train_20))
sorted_1 = not_sorted.sort("Distance")
genre_and_distances = Table().with_columns("Genre", sorted_1.column("Genre"),"Distance",sorted_1.column("Distance"))
genre_and_distances


# Computed the 5-nearest neighbors classification of the first movie in the test set.  That is, decide on its genre by finding the most common genre among its 5 nearest neighbors, according to the distances you've calculated.  Then check whether your classifier chose the right genre.  (Depending on the features you chose, your classifier might not get this movie right, and that's okay.)
# In[123]:
genre_and_distances.sort("Distance",descending=True).take(np.arange(0,5))

# In[130]:


# Sets my_assigned_genre to the most common genre among these.
my_assigned_genre = most_common("Genre", genre_and_distances.take(np.arange(0,5)))


print("The assigned genre, {}, was{}correct.".format(my_assigned_genre, " " if my_assigned_genre_was_correct else " not "))

# ## 3.2. A classifier function
# In[132]:


def classify(test_row, train_rows, train_classes, k):
    """Returns the most common class among k nearest neigbors to test_row."""
    distances = fast_distances(test_row, train_rows)
    not_sorted = train_movies.with_column("Distance",distances)
    sorted_1 = not_sorted.sort("Distance")
    genre_and_distances = Table().with_columns("Genre", sorted_1.column("Genre"),"Distance",sorted_1.column("Distance"))
    return most_common("Genre", genre_and_distances.take(np.arange(0,k)))

# Assigned `king_kong_genre` to the genre predicted by your classifier for the movie "king kong" in the test set, using **11 neighbors** and using your 20 features.

# In[134]:


not_sorted = train_movies.with_column("Distance",fast_distances(king_kong_features,train_20))
sorted_1 = not_sorted.sort("Distance")
genre_and_distances = Table().with_columns("Genre", sorted_1.column("Genre"),"Distance",sorted_1.column("Distance"))
genre_and_distances

king_kong_features = test_movies.where("Title",are.equal_to("king kong")).select(my_20_features).row(0)
king_kong_features

# The staff solution first defined a row object called king_kong_features.
king_kong_features = test_movies.where("Title",are.equal_to("king kong")).select(my_20_features).row(0)
king_kong_genre = classify(king_kong_features, train_20, train_movies.column("Genre"), 11)
king_kong_genre

# Finally, when we evaluate our classifier, it will be useful to have a classification function that is specialized to use a fixed training set and a fixed value of `k`.

# Created a classification function that takes as its argument a row containing your 20 features and classifies that row using the 11-nearest neighbors algorithm with `train_20` as its training set.

# In[ ]:


def classify_one_argument(row):
    return classify(row, train_20, train_movies.column("Genre"), 11)

classify_one_argument(test_20.row(0))


# ## 3.3. Evaluating your classifier

# In[ 121]:

test_guesses = test_20.apply(classify_one_argument)
proportion_correct = np.count_nonzero(test_guesses == test_movies.column("Genre")) / test_movies.num_rows
proportion_correct

train_20
# In[158]:

train_movies.select(np.append(my_20_features,staff_features))

# In[182]:

# To start you off, here's a list of possibly-useful features:
# FIXME: change these example features
staff_features = make_array("come", "do", "have", "heart", "make", "never", "now", "wanna", "with", "yo")

train_staff = train_movies.select(staff_features)
test_staff = test_movies.select(staff_features)
new_train = train_movies.select(np.append(my_20_features,staff_features))
test_30 = test_movies.select(np.append(my_20_features,staff_features))
def another_classifier(row):
    return classify(row, train_20, train_movies.column("Genre"), 3)


# In[181]:


new_test_guesses = test_20.apply(another_classifier)
new_proportion_correct = np.count_nonzero(new_test_guesses == test_movies.column("Genre")) / test_movies.num_rows
new_proportion_correct


# Briefly describe what you tried to improve your classifier. As long as you put in some effort to improving your classifier and describe what you have done, you will receive full credit for this problem.

# Original prediction 73.
# 
# I first tried to manipulate the values of K, and when I increased it to 20 my prediction dropped to 67.5 and when I decreased it to 7 my prediction increased to 78. When that did not help out my predication, I then tried to append the staff features to my original 20, and that ended up not helping at all either, lowering my prediction to 46. Lastly, I just ended up using the staff variables and there was again no change in my prediction and that brought my prediction down to 59. Since the only increase I got was by dropping the K value to 7 I ended up dropping it to 3 and it again increased my prediction to the same value as 7's 78.

# Congratulations: you're done with the required portion of the project! Time to submit.

# In[183]:


_ = ok.submit()

# For your convenience, you can run this cell to run all the tests at once!
import os
print("Running all tests...")
_ = [ok.grade(q[:-3]) for q in os.listdir("tests") if q.startswith('q')]
print("Finished running all tests.")

