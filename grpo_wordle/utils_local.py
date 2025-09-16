"""
Wordle Util Functions for Playing WORDLE (Local Version)

This module contains utils functions for prompting an LLM to play WORDLE
Modified to work with local Hugging Face models instead of Predibase APIs
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tabulate import tabulate


SYSTEM_PROMPT = """
You are playing Wordle, a word-guessing game.

### Game Rules:
- You have **6 tries** to guess a secret **5-letter** word.
- Each guess must be a valid **5-letter English word**.
- After each guess, you will receive feedback indicating how close your guess was.

### Feedback Format:
Each letter in your guess will receive one of three symbols:
1. ✓ : The letter is in the word and in the CORRECT position.
2. - : The letter is in the word but in the WRONG position.
3. x : The letter is NOT in the word.

### Example:
Secret Word: BRISK

Guess 1: STORM → Feedback: S(-) T(x) O(x) R(-) M(x)
Guess 2: BRAVE → Feedback: B(✓) R(✓) A(x) V(x) E(x)
Guess 3: BRISK → Feedback: B(✓) R(✓) I(✓) S(✓) K(✓)

### Response Format:
Think through the problem and feedback step by step. Make sure to first add your step by step thought process within <think> </think> tags. Then, return your guessed word in the following format: <guess> guessed-word </guess>.
"""


class LetterFeedback(Enum):
    CORRECT = "✓"
    WRONG_POS = "-"
    WRONG_LETTER = "x"


def get_feedback(guess: str, secret_word: str) -> List[LetterFeedback]:
    """Generate feedback for a guess against the secret word."""
    valid_letters = set(secret_word)
    feedback = []
    for letter, secret_letter in zip(guess, secret_word):
        if letter == secret_letter:
            feedback.append(LetterFeedback.CORRECT)
        elif letter in valid_letters:
            feedback.append(LetterFeedback.WRONG_POS)
        else:
            feedback.append(LetterFeedback.WRONG_LETTER)
    return feedback


@dataclass
class GuessWithFeedback:
    guess: str
    feedback: List[LetterFeedback]

    def __repr__(self) -> str:
        feedback_str = " ".join(f"{letter}({fb.value})" for letter, fb in zip(self.guess, self.feedback))
        return f"{self.guess} → Feedback: {feedback_str}"

    @staticmethod
    def from_secret(guess: str, secret: str) -> "GuessWithFeedback":
        return GuessWithFeedback(guess, get_feedback(guess, secret))


def render_user_prompt(past_guesses: List[GuessWithFeedback]) -> str:
    """Create user prompt with past guesses."""
    prompt = "Make a new 5-letter word guess."
    if past_guesses:
        prompt += "\n\nHere is some previous feedback:"
        for i, guess in enumerate(past_guesses):
            prompt += f"\nGuess {i+1}: {guess}"
    return prompt


def get_messages(past_guesses: List[GuessWithFeedback]):
    """Create message list for the model."""
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": render_user_prompt(past_guesses)
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
        }
    ]


def generate_local(
    model,
    tokenizer,
    messages: List[dict],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    verbose: bool = True
) -> str:
    """Generate completion using local model."""
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        continue_final_message=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new completion (remove the prompt)
    completion = generated_text[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
    
    if verbose:
        print(completion)
    
    return completion


def extract_guess(completion: str) -> str:
    """Extract the guess from model completion."""
    match = re.search(r"<guess>\s*([\s\S]*?)\s*</guess>", completion, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip().upper()


def next_turn(model, tokenizer, past_guesses: List[GuessWithFeedback], secret_word: str, verbose: bool = True):
    """Play one turn of Wordle."""
    messages = get_messages(past_guesses)
    completion = generate_local(model, tokenizer, messages, verbose=verbose)
    guess = extract_guess(completion)
    
    if not guess or len(guess) != 5:
        if verbose:
            print(f"Invalid guess extracted: '{guess}'")
        return None
    
    feedback = get_feedback(guess, secret_word)
    past_guesses.append(GuessWithFeedback(guess, feedback))
    
    if verbose:
        print("\n" + ("-" * 50))
        for past_guess in past_guesses:
            print(past_guess)
    
    if guess == secret_word:
        if verbose:
            print("🎉 SUCCESS 🎉")
        return True
    elif len(past_guesses) >= 6:
        if verbose:
            print("❌ Better luck next time... ❌")
        return False
    
    return None  # Game continues


def play_full_game(model, tokenizer, secret_word: str, max_guesses: int = 6, verbose: bool = False):
    """Play a full game of Wordle and return number of guesses and success status."""
    past_guesses = []
    
    for _ in range(max_guesses):
        result = next_turn(model, tokenizer, past_guesses, secret_word, verbose=verbose)
        if result is not None:  # Game ended
            return len(past_guesses), result
    
    # Should not reach here if next_turn works correctly
    return len(past_guesses), False


def compute_advantages(rewards: list):
    """Compute advantages from rewards (for GRPO context)."""
    rewards = np.array(rewards)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    if std_reward == 0:
        return [0] * len(rewards)

    advantages = (rewards - mean_reward) / std_reward
    return advantages.tolist()


def print_guesses_table(extracted_guesses, rewards):
    """Print formatted table of guesses and rewards."""
    advantages = compute_advantages(rewards)
    length = len(extracted_guesses)
    elems = list(zip(range(length), extracted_guesses, rewards, advantages))

    headers = ["Index", "Guess", "Reward", "Advantage"]
    table = tabulate(elems, headers=headers, tablefmt="grid").split("\n")
    for row in table:
        print(row)
