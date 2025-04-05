import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, Flatten, Dense, SpatialDropout1D, BatchNormalization, Dropout, SimpleRNN, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import GRU, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.regularizers import l2
import help_tar

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU for training!")
else:
    print("No GPU detected, using CPU.")

navec = help_tar.get_navec()

DATASET_PATH = "./dataset"
FILE_DIR_POEMS = os.path.join(DATASET_PATH, "poems")
FILE_DIR_PROSE = os.path.join(DATASET_PATH, "prose")

file_list_poems = os.listdir(FILE_DIR_POEMS)
file_list_prose = os.listdir(FILE_DIR_PROSE)
CLASS_LIST = list(set(file_list_poems))  #+ file_list_prose

all_texts = {}
for author in CLASS_LIST:
    all_texts[author] = ""
    for path in glob.glob(f'{FILE_DIR_POEMS}/{author}/*.txt'): #glob.glob(f'{FILE_DIR_PROSE}/{author}/*.txt') + 
        with open(path, 'r', errors='ignore') as f:
            text = f.read()
        all_texts[author] += ' ' + text.replace('\n', ' ')

# Tokenization
tokenizer = Tokenizer(num_words=15000, lower=True, split=' ')
tokenizer.fit_on_texts(all_texts.values())
seq_train = tokenizer.texts_to_sequences(all_texts.values())

def seq_split(sequence, win_size, step):
    return [sequence[i:i + win_size] for i in range(0, len(sequence) - win_size + 1, step)]

def seq_vectorize(seq_list, test_split, val_split, class_list, win_size, step):
    x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
    for class_item in class_list:
        cls = class_list.index(class_item)
        total_len = len(seq_list[cls])
        train_end = int(total_len * (1 - test_split - val_split))
        val_end = int(total_len * (1 - test_split))
        vectors_train = seq_split(seq_list[cls][:train_end], win_size, step)
        vectors_val = seq_split(seq_list[cls][train_end:val_end], win_size, step)
        vectors_test = seq_split(seq_list[cls][val_end:], win_size, step)
        x_train += vectors_train
        x_val += vectors_val
        x_test += vectors_test
        y_train += [cls] * len(vectors_train)
        y_val += [cls] * len(vectors_val)
        y_test += [cls] * len(vectors_test)
    return np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val), np.array(x_test), np.array(y_test)

x_train, y_train, x_val, y_val, x_test, y_test = seq_vectorize(seq_train, 0.2, 0.1, CLASS_LIST, 1000, 100)

def loadEmbedding():
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((15000, 300))
    for word, i in word_index.items():
        if i < 15000:
            embedding_vector = navec.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

model = Sequential()
model.add(Embedding(15000, 300, input_length=1000, weights=[loadEmbedding()], trainable=False))
model.add(SpatialDropout1D(0.2))
model.add(BatchNormalization())
model.add(Conv1D(128, kernel_size=5, activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(CLASS_LIST), activation='softmax'))
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) 

history = model.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test), callbacks=[lr_scheduler, early_stop])

y_pred = np.argmax(model.predict(x_val), axis=1)
acc = accuracy_score(y_val, y_pred)
print(f"Test Accuracy: {acc:.2%}")

conf_mat = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_LIST, yticklabels=CLASS_LIST)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

def predict_author(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = np.array(seq_split(seq[0], 1000, 100))
    preds = model.predict(seq)
    avg_pred = np.mean(preds, axis=0)
    plt.pie(avg_pred, labels=CLASS_LIST, autopct='%1.1f%%', startangle=140)
    plt.title("Author Prediction Probability")
    plt.show()
    return CLASS_LIST[np.argmax(avg_pred)]

with open('my.txt', 'rb') as file:
    result = file.read().decode('utf-8')

print(f"Predicted author: {predict_author(result)}")
