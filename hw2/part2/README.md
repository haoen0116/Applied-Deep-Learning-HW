# ADL Homework 2
This is the homework about the course of the Applied Deep Learning in National Taiwan University.

The task about homework 2 is to use the Language Sequence Classification to classify the news topic.
In this homework, we have to use the BERT language model to let the model know the content of the sentence, 
and use this model to predict the news topic of that sentence.

##How to run :
The env is in Python 3.7.2

Please install the requirements:
    
    pip install -r requirements.txt

Please download the model:
    
    bash download.sh

If you want to do [best.sh] please type the command below:
    
    bash best.sh [--testing.csv] [--output.csv]

If you want to do [strong.sh] please type the command below:
    
    bash strong.sh [--testing.csv] [--output.csv]
