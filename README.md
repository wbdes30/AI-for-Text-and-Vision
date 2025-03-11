# Assignments of AI for Text and Vision unit 
- **Assignment 1**:
  - Task 1: Return a mask for each channel that identifies the pixels whose intensity is above the given threshold. The mask of channel i is an array that has values 0 or 1.
  - Task 2: Return a histogram of the channel
  - Task 3: Implement a Keras neural model
    
- **Assignment 2**: Apply deep learning techniques for image classification. Tasks Overview:
  - Download a popular dataset for image classification, explore, and prepare it for the subsequent tasks.
  - Implement simple image classifiers and conduct an evaluation.
  - Implement Convolutional Network (ConvNet) and Pre-trained models (MobileNet) and conduct a more elaborate evaluation using confusion matrices from Keras datasets.
    
- **Assignment 3**: Work on a general answer selection task. Given a question and a list of sentences, the final goal is to predict which of these sentences from the list can be used as part of the answer to the question. Assignment 3 is divided into two parts.
  
  - Part 1: The data are in the file `train.csv`. Each row of the file consists of a question, a sentence text, and a label that indicates whether the sentence text is part of the answer to the question (1) or not (0).
    - Task 1: Find the top-N common NOUN in the questions. The function returns a list that is descendingly sorted in descending order of frequency
    - Task 2: Find the top-N common stem 2-grams and non-stem 2-grams, respectively, and decide which is more helpful to understand the common questions. Answer returns two lists (one for stem 2-grams, and the other one for non-stem 2-grams), and each is sorted in descending order of frequency
    - Task 3: Calculate the cosine similarity between one question and all its corresponding candidate sentences in the atext column, and check whether the sentence of the highest similarity has a label 1. Ultimately, report the proportion of questions can be accurately answered using the tf.idf feature.

  - Part 2: Study the similarity between the questions and the answers 
    - Task 1: Using Simple Siamese Neural Network - Contrastive Loss
    - Task 2: Using Simple Transformer Neural Network



