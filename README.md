<h1 style="text-align: center;">Agentic Reasonings via Reinforcement Fine Tuning</h1>

Agentic Reasonings via Reinforcement Fine Tuning (RFT) aims to apply RFT to improve reasoning capabilities and overall accuracies for an Agentic AI systems. This repo is a hands-on cookbook to tune and test your own models on OpenShift AI for you to follow.

<h2>Value Proposition for RFT</h2>

A list of key values it brings is:

1. In complex business scenarios, reasoning large language models (LLM) and RFT tuned agents have shown tremendous improvements on reasoning and accuracies in **multi-turn, multi-tools tasks**, greatly improving zero-shot capabilities.

2. Furthermore, there are no labeled data for multi-turn tasks since it extremely large sparse search space, thus prevents the usage of supervised fine-tuning, and RFT is the solution for scaling, requiring **no labeled data, no hand human feedback, and (sometimes) no hand-crafted reward functions**, simplifying adapting RL to new tasks.

The use of end of end (E2E) RFT on products with sparse search space has absolutely been the industry standard in Agentic AI products since early 2025, especially after the [DeepSeek GRPO paper](https://arxiv.org/abs/2402.03300).

Here are a list of recent products and research leveraging E2E RFT, all released in 2025:

1.  ChatGPT Agent (link to an [OpenAI Researcher's tweet](https://x.com/xikun_zhang_/status/1945895070269583554?s%3D46)), "@OpenAI RL diehards. You are probably tired of hearing about RL scaling. Me, too. But when I feel its power first-hand, its effectiveness and data efficiency still shock me and feel like magic 🪄".
2. Pokee AI (a [workflow agent](https://pokee.ai/home) product with 1000+ API integration), leverages a non-transformer RL model for tool calling.
3. Amazon. WEB AGENT-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning ([https://arxiv.org/pdf/2505.16421](https://arxiv.org/pdf/2505.16421)), first published on Thu, 22 May 2025. "We introduce WebAgent-R1, an end-to-end multi-turn RL framework that learns from online interactions with binary rewards", tested on the [AppWorld benchmark](https://github.com/stonybrooknlp/appworld).
4. Microsoft Research. Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning ([https://arxiv.org/pdf/2505.01441](https://arxiv.org/pdf/2505.01441)), "introduce(s) ARTIST, a framework unifying agentic reasoning, reinforcement learning, and tool integration for LLMs.“

<h2>Cookbooks</h2>

<h3>RFT WORDLE with GRPO</h3>

This repo provides a series of cookbooks for successfully leveraging RFT on different sort of tasks.

The `grpo_wordle` example leverages Huggingface's TRL (Transformer Reinforcement Learning) library to efficiently fine-tune an LLM to adapt to the task of [WORDLE](https://www.nytimes.com/games/wordle/index.html), without the need of labeled data.

By following along, you will discover how to define your own rule-based environment and train efficiently with TRL using methods like LoRA and PEFT with [WANDB](https://wandb.ai/home) as an option for observability. You will also run your own evaluation set to see the performance yourself.

<h3>RFT MCP servers and Function Calls</h3>
🛠️ This is a work in progress, aiming to reduce tool hallucinations for MCP servers and tool calling in general.