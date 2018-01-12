# Kaggle
A repository to work on a Kaggle Project

For a statewide science fair. Intended to be private and used as a bit of a shared workspace. 


# Config textblob, nltk, tensorflow

Textblob and nltk:

    pip3 install nltk
    pip3 install textblob
    
        
    # SSL doesn't trust the way to get resources, so we have to manually grab it and install the certificates. 
    
    pip3 install --trusted-host pypi.python.org textblob
    
    "/Applications/Python 3.6/Install Certificates.command"
    
    # Make sure the quotes are there
 
    
    
    
    
    # in your python console
    
    import nltk
    
    nltk.download()
   
    # Should open up window and you can choose what to download. Grab all of it.
    
    python3 -m textblob.download_corpora
    
    nltk.download('punkt')
    
Tensorflow:

This manually grabs the right version for our 3.6 interpreter.

    python3 -m pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py3-none-any.whl
    
    
    
 
# Data

data is too big to upload, but can be found here:

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
    
    
 
    
   
