# MLhackathon
Notebook for the ML hackathon 

# Intelligent Hangman Assistant (HMM + Q-Learning)

This project implements an intelligent agent that learns to play the game of Hangman. The agent's "brain" is a hybrid model that combines a statistical **Hidden Markov Model (HMM)** with a **Q-Learning Reinforcement Learning (RL) agent**.

The HMM provides a strong statistical "prior" by analyzing the structure of English, while the Q-Learning agent learns from game-playing experience to correct the HMM's general advice with specific, state-based knowledge.

## Project Structure

The project is contained within the `final.ipynb` notebook and is built around four core classes:

1.  **`HangmanHMM`**: A statistical model that learns letter probabilities from a text corpus.
2.  **`HangmanEnvironment`**: A simple, OpenAI Gym-style environment for the game of Hangman.
3.  **`QLearningAgent`**: The RL agent that learns to make decisions by combining HMM guidance and Q-Learning.
4.  **`HangmanEvaluator`**: A helper class to test the agent's performance on an unseen dataset.

---

## 1. Implementation Details & Design Choices

### HangmanHMM (The Statistical Guide)

* **Why an HMM?** Hangman is a game of statistical inference. The agent needs a baseline understanding of language. The HMM is a lightweight, efficient model trained on `corpus.txt` to learn which letters are most likely to appear given the current board state.
* **Implementation:**
    * **Models-by-Length:** The HMM builds a *separate* statistical model for each word length (e.g., all 5-letter words are modeled together). This is crucial, as the statistics for 4-letter words are very different from 10-letter words.
    * **Probabilities (Model State):** For each word length, the HMM calculates two key probabilities:
        1.  **Positional Probabilities (Emission):** The probability of a letter appearing at a specific position (e.g., 'e' is very common as the 4th letter of a 4-letter word).
        2.  **Transitional Probabilities (Transition):** The probability of one letter following another (e.g., 'Q' is almost always followed by 'U').
* **Role:** At every step of the game, the HMM provides a probability distribution over all unguessed letters, which is used to guide the RL agent.

### HangmanEnvironment (The Game)

* **Why?** To train an RL agent, we need a standardized environment that provides states, accepts actions, and returns rewards.
* **Implementation:** A standard environment with `step(action)` and `reset()` methods.
* **Reward Design:** The reward structure is designed to heavily penalize mistakes, as the agent only has 6 lives.
    * **Won Game:** `+100` (Strong positive terminal reward)
    * **Lost Game:** `-50` (Strong negative terminal reward)
    * **Correct Guess:** `+5` (Small positive reinforcement)
    * **Wrong Guess:** `-10` (Stronger penalty to discourage guessing)
    * **Repeated Guess:** `-2` (Small penalty to ensure efficiency)

### QLearningAgent (The "Brain")

* **Why Q-Learning?** Q-Learning is a tabular RL method that learns a "Q-value" for every `(state, action)` pair. This allows the agent to learn from experience. It can, for example, learn that while the HMM suggests 'E' for `_ _ _ _ E`, if it has already guessed 'E' and was wrong, it must override the HMM.
* **State Representation:**
    * **Design:** `f"{masked_word}:{guessed_letters_sorted}:{lives_left}"`
    * **Why:** This string representation is **fully Markovian**—it uniquely captures all information needed to make an optimal decision.
    * **Challenge:** This design creates an **enormous state space**. The agent will only ever visit a tiny fraction of all possible states, making the Q-table very **sparse**. This is the primary bottleneck of this approach.
* **Hybrid Policy (Exploitation):**
    * This is the core innovation. The agent's "best" action is not just the one with the highest Q-value. It's a hybrid score:
    * `score = (0.6 * Q_value) + (0.4 * HMM_probability * 10)`
    * This allows the agent to **combine learned experience (Q-value) with statistical guidance (HMM)**.
* **Guided Exploration (Exploration):**
    * We use a standard $\epsilon$-greedy policy with a decaying `epsilon`.
    * However, when the agent *chooses to explore* (with probability $\epsilon$), it doesn't pick a uniformly random letter. It makes a **weighted random choice** based on the HMM's probability distribution.
    * This is far more efficient, as "exploratory" guesses are still intelligent, probing high-probability letters first.

---

## 2. Training, Results & Discussion

### Training
The agent was trained for **5,000 episodes** using the `corpus.txt` as both the HMM training data and the RL environment's word list. The training graph clearly shows the average reward per episode trending upwards, confirming that the agent is successfully learning.



### Evaluation Results
The fully trained agent was then evaluated on **2,000 unseen words** from `test.txt`.

| Metric | Result |
| :--- | :--- |
| **Success Rate** | **25.40%** |
| Games Won | 508 / 2000 |
| Avg. Wrong Guesses | 5.45 |
| Total Repeated Guesses | 0 |
| **Final Score** | **-53942.0** |

The **0 repeated guesses** show that the `-2` reward penalty was 100% effective. The 25.4% win rate is respectable, given the massive state space and limited training.



The win rate varies significantly by word length, suggesting the HMM's statistical models are stronger for certain lengths.

### Why not a Deep Q-Network (DQN)?

A DQN, which uses a neural network to *approximate* the Q-function, was also prototyped for this problem.

* **The Theory:** A DQN *should* be superior because it can **generalize** across similar states. A tabular Q-table treats `A_PLE:B,C:5` and `A_PLE:K,D:5` as two completely different, unrelated states. A DQN could recognize their similarity and apply learned knowledge from one to the other.

* **The Reality:** In this project, the DQN's performance was *lower* than the tabular Q-Learning agent. The reason is the **nature and scale of the dataset**.
    1.  **Limited Training Data:** 5,000 episodes is a very small number for training a deep neural network. The DQN did not have enough data to learn the complex patterns and likely failed to converge.
    2.  **Sparsity vs. Memorization:** The tabular method, while suffering from state-space sparsity, was able to *memorize* the Q-values for the few common states it encountered. The DQN, trying to find a general function, struggled to do even this with the limited data.

Given more time and compute (e.g., 100,000+ training episodes), the DQN would almost certainly learn a superior policy and surpass the tabular agent.

---

## 3. Future Improvements

If I had another week, I would focus on overcoming the state-space limitation:

1.  **Massively Scale DQN Training:** Train the DQN for 100,000 to 500,000 episodes to allow the network to converge properly.
2.  **HMM as a Feature:** Feed the 26-value probability vector from the HMM *directly into the DQN* as part of the state. This would allow the network to learn *how much to trust* the HMM's advice, rather than using a fixed 60/40 split.
3.  **Better Language Model:** Replace the HMM with a more powerful n-gram model or a small, pre-trained character-level RNN to provide even better statistical guesses.

## 4. How to Run

1.  Upload the `corpus.txt` and `test.txt` files to your environment.
2.  Run all cells in the `final.ipynb` notebook sequentially.
3.  The notebook will:
    * Train the HMM (saved as `hmm_model.pkl`).
    * Train the RL Agent (saved as `agent_model.pkl`).
    * Plot the training progress.
    * Run the final evaluation and save the results (`evaluation_results.json`).
    * Plot the win rate by word length.
    * Download all the saved models, results, and charts to your local machine.
