import pandas as pd
import numpy as np
import keras
import tensorflow
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

# Global variables to store the trained model and preprocessing objects
model = None
tokenizer = None
label_encoder = None
max_length = None

def train_model():
    global tokenizer, label_encoder, max_length, model

    data = pd.read_csv("text_emotions/train.txt", sep=';')
    data.columns = ["Text", "Emotions"]

    texts = data["Text"].tolist()
    labels = data["Emotions"].tolist()

    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    max_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    # Encode the string labels to integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # One-hot encode the labels
    one_hot_labels = keras.utils.to_categorical(labels)

    # Split the data into training and testing sets
    xtrain, xtest, ytrain, ytest = train_test_split(padded_sequences,
                                                    one_hot_labels,
                                                    test_size=0.2)

    # Define the model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,
                        output_dim=128, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=len(one_hot_labels[0]), activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(xtrain, ytrain, epochs=10, batch_size=32, validation_data=(xtest, ytest))

def preprocess_and_classify(input_text):
    global tokenizer, label_encoder, max_length, model

   
    # Preprocess the input text
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
    prediction = model.predict(padded_input_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])

    return predicted_label

if __name__ == "__main__":
    if model is None:
        train_model()  # Train the model if not already trained

    while True:
        input_text = input("Enter a text (or 'no' to end): ")

        if input_text.lower() == "yes":
            break

        predicted_label = preprocess_and_classify(input_text)
        print("Predicted emotion:", predicted_label)

  
