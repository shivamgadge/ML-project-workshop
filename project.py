#Step1: Import the necessary libraries
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

import gym

class ChatbotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, messages):
        self.action_space = gym.spaces.Discrete(2) # Two possible actions: respond with a fixed message or ask a follow-up question
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(messages),)) # Binary vector representing the user's message
        self.messages = messages
        self.state = None

    def step(self, action):
        if action == 0: # Respond with a fixed message line
            response = "Thank you for your message. Our team will get back to you soon."
            reward = 1
            done = True
        else: # Ask a follow-up question from user
            response = "Can you provide more information about your request?"
            reward = 0
            done = False

        return self.state, reward, done, {'response': response}

    def reset(self):
        self.state = np.zeros(len(self.messages))
        return self.state
        
    
    # this step will Collect and preprocess data
messages = ['How can I help you?', 'Can you provide more information?', 'Thank you for your message.']
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load the chat logs
chat_logs = []
with open('chat_log.txt', 'r') as f:
    for line in f:
        line = line.strip().lower()
        if line:
            chat_logs.append(line)

# Tokenize and preprocess the chat messages
tokenized_messages = []
for message in messages:
    tokens = word_tokenize(message.lower())
    tokens = lemmatizer.lemmatize(token) #[for token in tokens if token is not in stop_words]
    tokenized_messages.append(' '.join(tokens))

# Tokenize and preprocess chat logs through the below code.
tokenized_logs = []
for log in chat_logs:
    tokens = word_tokenize(log)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token is not  stop_words]
    tokenized_logs.append(' '.join(tokens))

# Create a tokenizer and fit it on tokenized messages.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokenized_messages)

# Convert tokenized log to sequences
sequences = tokenizer.texts_to_sequences(tokenized_logs)

# Pad sequences to ensure each sequences have the same length.
max_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
#Step 4: This step will create a chatbot environment, an agent, and train the agent:

# This step will create a chatbot environment and agent.
env = ChatbotEnv(tokenized_messages)
agent = ChatbotAgent(env, tokenizer)

# Train the agent with the a train() function.
agent.train(episodes=1000)
#Stop 5: Finally, we will test our chatbot.



# Test the created chatbot
while True:
    user_input1 = input("User: ")
    tokens = word_tokenize(user_input1.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    tokenized_input = ' '.join(tokens)
    sequence = tokenizer.texts_to_sequences([tokenized_input])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    action = agent.act(padded_sequence[0], None, None)
    response = env.step(action)[3]['response']
    print("Chatbot : " + # Test the chatbot
while True:
    user_input1 = input("User: ")
    tokens = word_tokenize(user_input1.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    tokenized_input = ' '.join(tokens)
    sequence = tokenizer.texts_to_sequences([tokenized_input])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    action = agent.act(padded_sequence[0], None, None)
    response = env.step(action)[3]['response']
    print("Chatbot response: " + response))

        
        
        
        
        
        
