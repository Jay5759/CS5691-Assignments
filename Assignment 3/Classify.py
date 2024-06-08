import os
import glob
import pickle

def load_data(folder_name):
    emails_list = [] # for storing list of emails
    for i, filename in enumerate(sorted(glob.glob(os.path.join(folder_name, "email*.txt"))), start=1):  # get that particular file path
        with open(filename, "r", encoding='utf-8', errors='ignore') as fileObject:
            email = fileObject.read().lower()  # read the file and convert to lower case
            emails_list.append(email)  # append the email to a list
    return emails_list

def classify():
    test_folder = 'test' # load the list of emails in the test folder
    file_model = 'model.object'
    file_cv = 'count_vectorizer.object'

    if os.path.isfile(file_model) and os.path.isfile(file_cv):
        loaded_model = pickle.load(open(file_model, 'rb'))  # load the SVM classifier
        cv_model = pickle.load(open(file_cv, 'rb'))  # load the feature extractor
        x_test = load_data(test_folder)  # Loading test emails from test folder 
        test = cv_model.transform(x_test)  # extract the features, here frequency of words
        predictions = loaded_model.predict(test)
        print(predictions)  # print predictions
        with open("output.txt", "w") as output:  # write predictions to a file
            output.write(str(predictions))
    else:
        print('Please copy '+file_model+' and '+file_cv+' files from zip into this directory before ' 'running Classify.py ')

classify()