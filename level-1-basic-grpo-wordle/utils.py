import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Callable, Optional, Any
import importlib
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# Safe dotenv import
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv() -> None:  # fallback no-op
        return None

load_dotenv()

base_model_id = "Qwen/Qwen2.5-7B-Instruct"

_tokenizer: Optional[Any] = None
_model: Optional[Any] = None


def _get_transformers():
    return importlib.import_module("transformers")


def _get_torch():
    return importlib.import_module("torch")


def _get_datasets():
    return importlib.import_module("datasets")


def _get_trl():
    return importlib.import_module("trl")


def _get_tabulate():
    return importlib.import_module("tabulate")


def _get_tokenizer(model_id: str = base_model_id):
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    transformers = _get_transformers()
    AutoTokenizer = getattr(transformers, "AutoTokenizer")
    _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return _tokenizer


def _get_model(model_id: str = base_model_id):
    global _model
    if _model is not None:
        return _model
    torch = _get_torch()
    transformers = _get_transformers()
    AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    return _model


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
    prompt = "Make a new 5-letter word guess."
    if past_guesses:
        prompt += "\n\nHere is some previous feedback:"
        for i, guess in enumerate(past_guesses):
            prompt += f"\nGuess {i+1}: {guess}"
    return prompt


def get_messages(past_guesses: List[GuessWithFeedback]):
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": render_user_prompt(past_guesses),
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>",
        },
    ]


def render_prompt(past_guesses: List[GuessWithFeedback]):
    tokenizer = _get_tokenizer()
    messages = get_messages(past_guesses)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, continue_final_message=True
    )


def extract_guess(completion: str) -> str:
    match = re.search(r"<guess>\s*([\s\S]*?)\s*<\/guess>", completion, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip().upper()


def compute_advantages(rewards: list):
    import numpy as np  # local import to avoid top-level dependency

    rewards = np.array(rewards)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards) if (rewards.size > 0) else 0.0

    if std_reward == 0:
        return [0] * len(rewards)

    advantages = (rewards - mean_reward) / std_reward
    return advantages.tolist()


def print_guesses_table(extracted_guesses, rewards):
    tabulate = getattr(_get_tabulate(), "tabulate")

    advantages = compute_advantages(rewards)
    length = len(extracted_guesses)
    elems = list(zip(range(length), extracted_guesses, rewards, advantages))

    headers = ["Index", "Guess", "Reward", "Advantage"]
    table = tabulate(elems, headers=headers, tablefmt="grid").split("\n")
    for row in table:
        print(row)


# === Local generation helpers (replacement for Predibase/OpenAI) ===

def _messages_to_prompt(messages: List[dict]) -> str:
    tokenizer = _get_tokenizer()
    return tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)


def generate_stream(
    prompt: str,
    adapter_id: str = "",
    temperature: float = 0.7,
    max_tokens: int = 256,
    stream: bool = True,
) -> str:
    _ = adapter_id  # keep signature compatibility

    torch = _get_torch()
    TextStreamer = getattr(_get_transformers(), "TextStreamer")

    model = _get_model()
    tokenizer = _get_tokenizer()
    device = model.device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=max(temperature, 1e-5) if do_sample else None,
        top_p=0.95 if do_sample else None,
        streamer=streamer,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        output_ids = model.generate(**inputs, **{k: v for k, v in gen_kwargs.items() if v is not None})

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    if stream:
        print()
    return text


def generate(
    messages: List[dict],
    adapter_id: str = "",
    num_guesses: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> List[str]:
    _ = adapter_id  # keep signature compatibility

    prompt = _messages_to_prompt(messages)
    outputs: List[str] = []
    for _i in range(max(1, num_guesses)):
        outputs.append(
            generate_stream(
                prompt,
                adapter_id=adapter_id,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
        )
    return outputs


def next_turn(past_guesses: List[GuessWithFeedback], secret_word: str, adapter_id: str = ""):
    _ = adapter_id  # keep signature compatibility
    prompt = render_prompt(past_guesses)
    completion = generate_stream(prompt)
    guess = extract_guess(completion)

    feedback = get_feedback(guess, secret_word)
    past_guesses.append(GuessWithFeedback(guess, feedback))
    print("\n\n")
    print(("-" * 100) + "\n")
    for past_guess in past_guesses:
        print(past_guess)

    if guess == secret_word:
        print("🎉 SUCCESS 🎉")
    elif len(past_guesses) >= 6:
        print("❌ better luck next time... ❌")


# === TRL / GRPO integration ===

def make_wordle_reward(secret_word: str) -> Callable[[List[str]], List[float]]:
    """Create a reward function for Wordle-like outputs.

    Reward = number of correct-position letters in the extracted <guess> (0..5),
    with a small bonus for exact match.
    """

    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        _ = kwargs  # unused
        rewards: List[float] = []
        for completion in completions:
            guess = extract_guess(completion)
            if len(guess) != 5 or not guess.isalpha():
                rewards.append(-1.0)
                continue
            guess = guess.upper()
            correct_positions = sum(1 for g, s in zip(guess, secret_word) if g == s)
            exact_bonus = 1.0 if guess == secret_word else 0.0
            rewards.append(float(correct_positions) + exact_bonus)
        return rewards

    return reward_fn


def build_wordle_dataset(past_guesses: List[GuessWithFeedback], repeat: int = 128):
    """Create a minimal dataset with a single prompt repeated for GRPO online training.

    GRPO will sample completions online; we only need prompts.
    """
    Dataset = getattr(_get_datasets(), "Dataset")

    prompt = render_prompt(past_guesses)
    return Dataset.from_list([{"prompt": prompt} for _ in range(repeat)])


def train_wordle_grpo(
    secret_word: str,
    past_guesses: Optional[List[GuessWithFeedback]] = None,
    model_id: str = base_model_id,
    output_dir: str = "wordle-grpo-output",
    max_steps: int = 100,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    num_generations_per_prompt: int = 8,
    max_prompt_length: int = 512,
    max_completion_length: int = 64,
    beta: float = 0.0,
    learning_rate: float = 1e-6,
    seed: int = 42,
    generation_batch_size: Optional[int] = None,
):
    """Run GRPO training to improve Wordle guessing behavior.

    This sets beta=0.0 by default to avoid loading a separate reference model, per TRL docs.
    See: https://huggingface.co/docs/trl/main/en/grpo_trainer
    """
    try:
        trl_module = _get_trl()
        GRPOConfig = getattr(trl_module, "GRPOConfig")
        GRPOTrainer = getattr(trl_module, "GRPOTrainer")
    except Exception:
        # Fallback: try internal path (older/newer layouts). If it fails, prompt upgrade.
        try:
            grpo_mod = importlib.import_module("trl.trainer.grpo")
            GRPOConfig = getattr(grpo_mod, "GRPOConfig")
            GRPOTrainer = getattr(grpo_mod, "GRPOTrainer")
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "TRL installation does not expose GRPOConfig/GRPOTrainer. "
                "Please upgrade TRL to a version that includes GRPO (e.g., pip install -U 'trl>=0.21.0')."
            ) from e

    if past_guesses is None:
        past_guesses = []

    if num_generations_per_prompt < 2:
        raise ValueError("num_generations_per_prompt must be >= 2 for GRPO.")

    # Default generation batch size to num_generations to satisfy divisibility constraint
    if generation_batch_size is None:
        generation_batch_size = num_generations_per_prompt
    elif generation_batch_size % num_generations_per_prompt != 0:
        # Round up to the next multiple to satisfy config constraint
        generation_batch_size = (
            (generation_batch_size // num_generations_per_prompt + 1)
            * num_generations_per_prompt
        )

    dataset = build_wordle_dataset(past_guesses)

    config = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_generations=num_generations_per_prompt,
        generation_batch_size=generation_batch_size,
        beta=beta,
        seed=seed,
        log_completions=True,
        num_completions_to_print=3,
        loss_type="bnpo",
        mask_truncated_completions=True,
        scale_rewards=False,
    )

    reward_fn = make_wordle_reward(secret_word)

    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=dataset,
    )

    trainer.train()

    return trainer
