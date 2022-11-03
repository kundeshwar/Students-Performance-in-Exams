#!/usr/bin/env python
# coding: utf-8

# # Students Performance in Exams
# 
# This data set consists of the marks secured by the students in various subjects. We will perform different operation on this data set . We will be using different pythons labrary for analysis data. 
# In this data set we will learn about how to handle data and also data cleaning process.data cleaning is the process of detecting and removing corrupt or inaccurate records from a record set, table, or database and refers to identifying incomplete, incorrect, inaccurate or irrelevant parts of the data and then replacing, modifying, or deleting the dirty or coarse data.
# In our data set consist of 1000 rows and 8 columns, so we have to handle this all rows and columns with systematic approach.
# This data set consist of marks of student got in various subject like math, writing, reading etc. also consist of race/ethnicity, also consist of gender of student like female and male.
# Also we will see data transformation is the process of converting data from one format or structure into another format or structure. we will see about some data modeling means analyze data requirements needed to support the business processes within the scope of corresponding information systems in organizations.

# ## Downloading the Dataset
# 
# In this step we will do data download from kaggle online website and read this data.

# In[242]:


get_ipython().system('pip install jovian opendatasets --upgrade --quiet')


# Let's begin by downloading the data, and listing the files within the dataset.

# In[243]:


# Change this
dataset_url = 'https://www.kaggle.com/datasets/whenamancodes/students-performance-in-exams' 


# In[244]:


import opendatasets as od
od.download(dataset_url)


# The dataset has been downloaded and extracted.

# In[245]:


# Change this
data_dir = './students-performance-in-exams'


# In[246]:


import os
os.listdir(data_dir)


# Let us save and upload our work to Jovian before continuing.

# In[247]:


project_name = "students-performance-in-exams" # change this (use lowercase letters and hyphens only)


# In[248]:


get_ipython().system('pip install jovian --upgrade -q')


# In[249]:


import jovian


# In[250]:


jovian.commit(project="kundeshwar_students_performance_in_exams")


# ## Data Preparation and Cleaning
# 
# In this step we will prepare data in useful manner and also clean data by using pandas labrary.n this data set we will learn about how to handle data and also data cleaning process.data cleaning is the process of detecting and removing corrupt or inaccurate records from a record set, table, or database and refers to identifying incomplete, incorrect, inaccurate or irrelevant parts of the data and then replacing, modifying, or deleting the dirty or coarse data.
# 
# 

# In[251]:


import pandas as pd
contend = pd.read_csv(data_dir + "/exams.csv")
contend


# In[252]:


contend.info()


# In[253]:


contend.head(10)


# In[254]:


contend.tail(10)


# In[255]:


contend.sample


# In[256]:


contend.describe()


# In[257]:


contend.shape


# In[258]:


contend["math score"]


# In[259]:


contend.at[245, "writing score"]


# In[260]:


contend.loc[45]


# In[261]:


contend["total_marks_in_all_sub"] = contend["math score"]+contend["writing score"]+contend["reading score"]
contend


# In[262]:


contend["percentage"] = (contend["total_marks_in_all_sub"]*100)/300
contend


# In[263]:


contend.sort_values("percentage",ascending=False).head(10)


# In[264]:


contend.sort_values("math score").head(10)


# In[265]:


b = "https://www.kaggle.com/datasets/devansodariya/student-performance-data"
import opendatasets as od
od.download(b)


# In[266]:


c = "./student-performance-data"
import os
os.listdir(c)


# In[267]:


total_info_student = pd.read_csv(c + "/student_data.csv")
total_info_student


# In[268]:


total_info_student.rename(columns = {'sex':'gender'}, inplace = True)
total_info_student


# In[269]:


frames = [contend, total_info_student]
result = pd.concat(frames)
result


# In[270]:


type(contend["math score"])


# In[271]:


contend.percentage.mean()


# In[272]:


b = (contend.percentage.sum())/1395
b


# In[273]:


contend_math = contend.groupby('gender')[['math score', 'writing score', 'reading score']].sum()
contend_math


# In[274]:


import jovian


# In[275]:


jovian.commit()


# ## Exploratory Analysis and Visualization
# 
# In this step we will analysis and visualization of data by using matplotlib and seaborn labrary. data visualization means is a particularly efficient way of communicating when the data or information is numerous as for example a time series.
# Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. 
# Matplotlib makes easy things easy and hard things possible.
# The seaborn library is widely used among data analysts, the galaxy of plots it contains provides the best possible representation of our data.
# 
# 

# Let's begin by importing`matplotlib.pyplot` and `seaborn`.

# In[276]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# we will convert some columns value into numapy array as follows.

# In[277]:


import numpy as np
math_score = np.array(contend["math score"].head(100))
writing_score = np.array(contend["writing score"].head(100))
reading_score = np.array(contend["reading score"].head(100))
percenatge = np.array(contend["percentage"].head(100))


# In[278]:


math_score
writing_score
reading_score
percenatge 
contend.describe()


# Firstly we will do simple line plot and pie chart between marks of student in math and writing.
# A line chart or line graph or curve chart is a type of chart which displays information as a series of data points called 'markers' connected by straight line segments.

# In[279]:


#In this graph we will show comparison between two different marks 
marks = range(0,100)
plt.plot(marks, math_score, marker="+")
plt.plot(marks, writing_score, marker="*")
plt.ylabel("MARKS(out of 100)", fontsize=(25))
plt.xlabel("SCORE IN SUBJECT", fontsize=(25))
plt.title("COMPARISION BETWEEN TWO MARKS")
plt.legend(['MATH', 'WRITING']);
#can you see that in this simple line plot describe difference between  marks in math and writing. 
#in this whole graph we can see two different line one is orange color and another is blue color both are separte marks of subject. 


# In[280]:


#In this graph we will analysis top 10 student percentage and there group
a = contend.sort_values("percentage",ascending=False).head(10)
plt.pie(a["percentage"], labels=a["race/ethnicity"])
plt.title("PERCENTAGE(top ten student) WITH THERE RACE/ETHNICITY");
#Below graph is pie chart
#This graph is showing different race/ethnicity with top 10 student with different color


# In this step we will make barchart of different subject marks.
# A bar chart or bar graph is a chart or graph that presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent. 
# The bars can be plotted vertically or horizontally.

# In[281]:


#we will analysis only first 10 data for the simplicity
#IN BELOW GRAPH WE WILL ANALYSIS MATH MARKS OF FIRST 10 STUDEND WITH GENDER
import numpy as np
sns.barplot(contend["gender"].head(10), contend["math score"].head(10))
plt.title("HOW SCORE MORE BETWEEN MALE AND FEMALE IN MATH SUBJECT")
plt.legend(['MATH'])
plt.show();


# In[282]:


#we will analysis only first 10 data for the simplicity
#IN BELOW GRAPH WE WILL ANALYSIS writing MARKS OF FIRST 10 STUDEND WITH GENDER
import numpy as np
sns.barplot(contend["gender"].head(10), contend["writing score"].head(10))
plt.title("HOW SCORE MORE BETWEEN MALE AND FEMALE IN writing SUBJECT")
plt.show();


# In[283]:


#we will analysis only first 10 data for the simplicity
#IN BELOW GRAPH WE WILL ANALYSIS  OF RACE/ETHNICITY FIRST 10 STUDEND WITH GENDER
sns.barplot(contend["race/ethnicity"].head(100), contend["percentage"].head(100), hue=contend["gender"].head(100))
plt.title("HOW PERCENTAGE CHANGES WITH RACE/ETHNICITY AND GENDER")
plt.show();
#And also we can seen how many female and male come in group [A,B,C,D]


# we will plot histogram between marks of student.
# A histogram is an approximate representation of the distribution of numerical data. The term was first introduced by Karl Pearson. 
# To construct a histogram, the first step is to bin the range of values that is, divide the entire range of values into a series of intervals and then count how many values fall into each interval.

# In[284]:


#In this graph we will plot histogram of marks and math score.
#we can see that on x-axis we put marks from the range 0 to 100 and in the y-axis we put math score from range 20 to 100
plt.hist(math_score,color="b",edgecolor="black",rwidth=0.8)
plt.ylabel("MARKS(out of 100)", fontsize=(25))
plt.xlabel("MATH SCORE", fontsize=(25))
plt.title("COMPARISION OF MATH MARKS");
# After observation you can see that this bins showing math marks distribution.
#by using this graph we can easily say math score distribution


# In[285]:


#In this graph we will plot histogram of marks and writing score.
#we can see that on x-axis we put marks from the range 0 to 100 and in the y-axis we put writing score from range 20 to 100
plt.hist(writing_score,color="b",edgecolor="black",rwidth=0.8)
plt.ylabel("MARKS(out of 100)", fontsize=(25))
plt.xlabel("writing SCORE", fontsize=(25))
plt.title("COMPARISION OF MATH MARKS");
# After observation you can see that this bins showing writing marks distribution.
#by using this graph we can easily say writing score distribution


# we will do count plot by using seaborn.
# we can create a countplot using the seaborn library and how the different parameters can be used to infer results from the features of our dataset.
# The countplot is used to represent the occurrence(counts) of the observation present in the categorical variable.
# It uses the concept of a bar chart for the visual depiction.

# In[286]:


#In this graph we will plot count-plot of gender of first 100 student.
#this is plot which seprate number male and female student in top 100
sns.countplot(x=contend["gender"].head(100))
plt.title("count male and female student");


# In[287]:


#In this graph we will plot count-plot of race of first 100 student.
#we can see In this count plot we can divide race/ethnicity of student of top 100 student
sns.countplot(x=contend["race/ethnicity"].head(100))
plt.title("count race/ethnicity of student")


# Let us save and upload our work to Jovian before continuing

# In[288]:


import jovian


# In[289]:


jovian.commit()


# ## Asking and Answering Questions
# 
# IF YOU HAVE CURRY ABOUT DATA AS FOLLOWS SO I TRY TO SOLVE FOLLOWING SOME QUESTION.
# 
# 

# #### Q1: Can you tell me how many student got 100% percentage?

# In[290]:


contend["percentage"]


# In[291]:


contend.sort_values("percentage",ascending=False).head(10)#if this data not showing total student who got 100% then increses value inside head()


# In[292]:


contend[contend.percentage == 100]


# In[293]:


#we will do count plot by using seaborn. 
#we can create a countplot using the seaborn library and how the different parameters can be used to infer results from the features of our dataset.
#we can see this plot gives how many male and female got 100 marks 
sns.countplot(x=contend[contend.percentage == 100].gender)
plt.title("count male and female student");


# #### Q2: Can you tell me how much type of group exit and how many student are in group(A, B, C, D, E), means I asking about count?

# In[294]:


contend["race/ethnicity"].unique()#this will show you how much type of group exit


# In[295]:


contend.groupby("race/ethnicity")[["gender"]].count()


# In[296]:


#In this graph we will plot count-plot of race/ethnicity.
#we can see In this count plot we can divide race/ethnicity of student.
sns.countplot(x=contend["race/ethnicity"])
plt.title("count race/ethnicity of student");
#we can separte total student by using race/ethnicity


# #### Q3: Can you tell me total male student and female student in our dataset?

# In[297]:


a = []
b = []
for i in contend["gender"]:
    if i == "male":
        a.append(i)
    elif i == "female":
        b.append(i)

print(f"{len(a)}Thie is total count of male")
print(f"{len(b)}Thie is total count of male")


# In[298]:


contend["gender"].count()


# In[299]:


contend.groupby("gender")[["race/ethnicity"]].count()


# In[300]:


#This is count plot which telling answer of our question by graphical from
#This count plot is very helpful to vaisalize data
sns.countplot(x=contend.gender)
plt.title("count male and female student");


# #### Q4: Can you tell me how many student are have percentage more than 50%?

# In[301]:


contend["percentage"]


# In[302]:


a = contend[contend.percentage>50].count()


# In[303]:


a.gender


# In[304]:


#by using this hist plot we can distribute percentage
#by seen this graph we can able tell me answer of our question.
#this hist plot is consist of 30 bins
plt.title("PERCENTAGE OF STUDENT")
sns.histplot(x=contend["percentage"], bins=30, kde=True);


# #### Q5: Can you plot scatter plot between math score contribution in percentage and also tell me how many student got out of marks in math subject.

# In[305]:


contend


# In[306]:


contend.sort_values("math score",ascending=False).head(14)


# In[307]:


#this is scatter plot which will give data anylsis by using small circul ball
#You can see that in below plot it directed divided data of percentage on the basis of math marks
sns.scatterplot(x=percenatge, y=marks, hue=math_score, s=100)
plt.ylabel("MARKS(out of 100)", fontsize=(25))
plt.xlabel("PERCENTAGE", fontsize=(25))
plt.title("COMPARISION(math score)");


# Let us save and upload our work to Jovian before continuing.

# In[308]:


import jovian


# In[309]:


jovian.commit()


# ## Inferences and Conclusion
# 
# In this dataset many data is in numerical from so it is very helpful to handel data. By using different plot like histogram, scatterplot etc. We can make complex data in simpler onces. 
# Python Library is really so much helpful to handle dataset.
# After this analysis we will say in short Pandas is a Software Library in Computer Programming and it is written for the Python Programming Language its work to do data analysis and manipulation.
# If we will analysis data so we must have pre-knowledge of some pythons library.
# After this basic analysis we can say python is very useful to data handling.
# If we have to analysis data we must knows pandas functions , matplotlib and seaborn library functions.
# And also we have to knows some math operations.

# In[310]:


import jovian


# In[ ]:


jovian.commit()


# ## References and Future Work
# 
# Yes I will very interested to do some extra work on this data. Following links are useful to get data set
# The next part of the article will attempt to dive further into the dataset and draw more meaningful conclusions using hypothesis testing.
# And i am also interested to  Data and information visualization is an interdisciplinary field that deals with the graphic representation of data and information. It is a particularly efficient way of communicating when the data or information is numerous as for example a time series.
# I have to improve my some skill like choosing the right content, planning marketing campaigns, or developing products, leading to better outcomes and customer satisfaction.
# I use this analysis for my future project it will help to get function in less time because you know it is very difficult to finding relevant function in short time so it is very helpful in my future data analysis journey.
# I heartily say thanks to this platform to give this opportunity.
# following some References which i use to analysis above data:
# 1.for data download ---https://www.kaggle.com/datasets
# 2.for pandas -------https://pandas.pydata.org/docs/
# 3.for matplotlib ------https://matplotlib.org/stable/index.html
# 4.for seaborn -------https://seaborn.pydata.org/

# In[240]:


import jovian


# In[241]:


jovian.commit()


# In[ ]:




