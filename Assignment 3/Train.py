import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import pickle

def load_data(folder_name):
    email_list = []  # Storing the list of emails
    email_label = [] # Storing the corresponding labels of emails

    spam_path = os.path.join(folder_name,'Spam') + os.path.sep
    ham_path = os.path.join(folder_name,'Ham') + os.path.sep

    for category in [spam_path,ham_path]: 
        for filename in os.listdir(category):
            file_path = os.path.join(category, filename)
            with open(file_path, 'r', encoding='latin1') as file:
                email_content = file.read()
                email_content = email_content.lower()
                email_list.append(email_content) # Add email to list of emails
                if "Ham" in category:  # depending on the folder we are reading the email from label them
                   email_label.append(0)
                elif "Spam" in category:
                   email_label.append(1)
    
    return np.array(email_list), np.array(email_label)  # return email array and corresponding labels


x_train, y_train = load_data('Train')
x_test, y_test = load_data('Validation')

frequency = {}
for email in x_train:
    words = email.split()
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1

threshold = 3000
high_freq_words = [word for word, freq in frequency.items() if freq > threshold]

x_train_filtered = []
for email in x_train:
    filtered_email = ' '.join(word for word in email.split() if word not in high_freq_words)
    x_train_filtered.append(filtered_email)
x_train_filtered = np.array(x_train_filtered)

x_test_filtered = []
for email in x_test:
    filtered_email = ' '.join(word for word in email.split() if word not in high_freq_words)
    x_test_filtered.append(filtered_email)
x_test_filtered = np.array(x_test_filtered)

# Plot frequency vs count
# word_counts = list(frequency.values())
# word_freqs = list(frequency.keys())

# plt.figure(figsize=(10, 6))
# plt.plot(word_counts)
# plt.xlabel('Word Index')
# plt.ylabel('Frequency')
# plt.title('Word Frequency vs. Count')
# plt.grid(True)
# plt.show()

# Remove high-frequency words from the frequency dictionary
for word in high_freq_words:
    del frequency[word]
d = len(frequency) # No. of words in dictionary

def accuracy(original, predicted):
    correct_predictions = sum(1 for o, p in zip(original, predicted) if p == o)
    return (correct_predictions / len(predicted)) * 100

def train_and_score_model(kernel, **kwargs):
    model = svm.SVC(kernel=kernel, **kwargs).fit(features, y_train)
    current_accuracy = accuracy(y_test, model.predict(test))
    print(f"{kernel.capitalize()} Kernel {kwargs}, Accuracy: {current_accuracy}")
    return current_accuracy, model

# Prepare features
cv = CountVectorizer()
features = cv.fit_transform(x_train_filtered)
test = cv.transform(x_test_filtered)

best_score = 0
best_model = None

# Linear kernel
current_accuracy, model = train_and_score_model('linear')
if current_accuracy > best_score:
    best_score = current_accuracy
    best_model = model

# Radial basis kernel with C=1,2,3
for C in range(1, 4):
    current_accuracy, model = train_and_score_model('rbf', C=C)
    if current_accuracy > best_score:
        best_score = current_accuracy
        best_model = model

# Polynomial kernel with degree=2,3,4
for deg in range(2, 5):
    current_accuracy, model = train_and_score_model('poly', degree=deg)
    if current_accuracy > best_score:
        best_score = current_accuracy
        best_model = model

filename_model = 'model.object'
pickle.dump(best_model, open(filename_model, 'wb'))  # store best model's object to disk
filename_cv = 'count_vectorizer.object'
pickle.dump(cv, open(filename_cv, 'wb'))  # store feature extractor's object to disk