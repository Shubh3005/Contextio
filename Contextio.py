import numpy as np
import random

# Function to load embeddings from a GloVe file
def load_glove_embeddings(file_path):
    """
    Load GloVe embeddings from a file.
    :param file_path: Path to the GloVe embeddings file.
    :return: Dictionary with words as keys and vectors as values.
    """
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vector
    return embeddings

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Main game function
def contexto_game(embedding_file):
    embeddings = load_glove_embeddings(embedding_file)
    words = list(embeddings.keys())
    
    # Randomly select a target word
    target_word = random.choice(words)
    target_vector = embeddings[target_word]
    
    print("Welcome to Contexto!")
    print("Guess the word based on its semantic similarity.")
    print("Type 'exit' to quit the game.\n")
    
    guesses = list(embeddings.keys())
    rankings = {word: cosine_similarity(target_vector, embeddings[word]) for word in guesses}
    ranked_words = sorted(rankings.items(), key=lambda item: item[1], reverse=True)
    
    # Extract the list of words in rank order
    ranked_word_list = [word for word, _ in ranked_words]
    
    attempts = 0
    while True:
        guess = input("Enter your guess: ").strip().lower()
        attempts += 1
        
        if guess == "exit":
            print(f"The target word was '{target_word}'. Goodbye!")
            break
        
        if guess not in embeddings:
            print("Word not in vocabulary. Try again.")
            continue
        print(target_word)
        if guess == target_word:
            print(f"Congratulations! You've guessed the word '{target_word}' in {attempts} attempts.")
            break
        
        # Provide rank feedback
        rank = ranked_word_list.index(guess) + 1
        print(f"Your guess '{guess}' is ranked {rank}. Keep trying!")

# Example usage
# Replace 'glove.6B.50d.txt' with the path to your GloVe file
embedding_file = "glove.6B.50d.txt"
contexto_game(embedding_file)
