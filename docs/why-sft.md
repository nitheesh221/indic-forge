# Why this project trains IndicTrans2 with SFT only

This note is for someone new to fine-tuning. It explains the one training method this repo uses, and why PPO, GRPO, DPO, and KTO are not part of the setup.

**Short answer:** IndicTrans2 already knows how to translate. We have sentence pairs (source text and the correct translation). Supervised fine-tuning (SFT) teaches the model those pairs directly. The other methods are for cases where there is no single correct answer and a separate judge has to score outputs. That is not our data, and it is not how IndicTrans2 was trained.

## What we are teaching the model

IndicTrans2 is a translation model from AI4Bharat. It is an encoder–decoder: one part reads the source sentence, the other writes the translation. It covers the 22 scheduled languages of India, in three directions: English → Indic, Indic → English, and Indic → Indic.

A fine-tune here means: take that model and teach it *our* sentences (for example public-service wording, or a mixed Indic-to-English file). Every training row looks like this:

```json
{"src": "Your Aadhaar OTP is valid for 10 minutes.", "tgt": "आपका आधार OTP 10 मिनट के लिए मान्य है।", "src_lang": "eng_Latn", "tgt_lang": "hin_Deva"}
```

`src` is the input. `tgt` is the translation we want. We already know the right answer.

## What SFT means, in one example

SFT means supervised fine-tuning. "Supervised" means a human (or a dataset) already wrote the correct output.

During training the model does not freely invent a translation and wait for a score. It is shown the Hindi sentence and asked, word by word (token by token): "given the English sentence and the Hindi words so far, what is the next Hindi word?"

If the next word should be `मान्य` and the model puts high probability on something else, the loss goes up. The optimizer then nudges the weights so that word becomes more likely next time. This loss is ordinary cross-entropy. Label smoothing (default `0.1` in this repo) slightly softens the target so the model does not become over-confident on one token.

That is the whole training objective. `--train_mode lora`, `qlora`, `full_lora`, and `full` only decide *which weights* receive that update. They do not change the lesson.

## Why SFT matches this repo

1. Our files are parallel text. Each line has one source and one reference translation.
2. The official IndicTrans2 LoRA script does the same thing. It uses Hugging Face `Seq2SeqTrainer`, whose training loss is cross-entropy on the target tokens. See [huggingface_interface/train_lora.py](https://github.com/AI4Bharat/IndicTrans2/blob/main/huggingface_interface/train_lora.py) in the IndicTrans2 repository.
3. The IndicTrans2 paper itself improves the model with a second **supervised** stage, not with reinforcement learning. After training on a large mix of parallel data, the authors fine-tune again on high-quality human translations (BPCC human data and NLLB seed). That second stage gained about **2.1 chrF++** for English → Indic and about **1 chrF++** for Indic → English on their main benchmarks. Citation: Gala et al., 2023, *IndicTrans2*, [arXiv:2305.16307](https://arxiv.org/abs/2305.16307), section "Impact of Sequential Training with Human Annotated Data".
4. A 2026 WMT system that adapts IndicTrans2 for low-resource English → Indic also uses LoRA supervised fine-tuning (AdamW, attention projections `q_proj` and `k_proj`), not a preference or reward method. Citation: DELAB-IIITM, WMT 2026, [anthology 2026.wmt-main.159](https://aclanthology.org/2026.wmt-main.159.pdf).

So both the people who built the model and later teams who adapted it used SFT when they had parallel sentences. This toolkit follows that.

## The other names, in plain language

These methods were built mainly to steer **chat models**, where many answers can be acceptable and nobody stored one official sentence.

| Name | What it asks the model to learn from | What you must already have |
| --- | --- | --- |
| **PPO** | "Make the reward model happier, but do not drift far from the original model." | A reward model, a second value model, and many sampled answers |
| **GRPO** | "Among the K translations you just wrote, imitate the ones with the higher reward." | K sampled translations per source, plus a reward score for each |
| **DPO** | "This translation was preferred over that one." | Pairs labeled chosen and rejected |
| **KTO** | "This single example is desirable, or this other one is not." | Good/bad labels, often built by damaging a real translation |

None of those files exist in this project. A row here is `src` + `tgt`, not "translation A beat translation B".

### PPO

PPO (Proximal Policy Optimization) is the algorithm used in the well-known InstructGPT / RLHF setup. The model generates an answer, a separate reward model scores it, and PPO updates the model with a penalty so it does not wander away from the starting checkpoint. Citation: Ouyang et al., 2022, *Training language models to follow instructions with human feedback*, [arXiv:2203.02155](https://arxiv.org/abs/2203.02155). Schulman et al., 2017 introduced PPO itself.

That pipeline exists because a chatbot prompt does not have one correct paragraph. A translation row in this repo does. Paying for a reward model and a value model to rediscover a Hindi sentence we already stored does not add a training signal we lack.

IndicTrans2’s paper never trains with PPO. Its reported gains come from more supervised bitext and then supervised fine-tuning on human translations.

### GRPO

GRPO (Group Relative Policy Optimization) drops PPO’s value model. For each source it samples a group of outputs, scores them, and treats "better than the group average" as the advantage. It was introduced for math reasoning models. Citation: Shao et al., 2024, *DeepSeekMath*, [arXiv:2402.03300](https://arxiv.org/abs/2402.03300).

A 2026 paper does apply GRPO to an encoder–decoder translator (NLLB, not IndicTrans2). It helps when fine-tuning has **no reference translation**: the reward is reference-free (LaBSE + COMET-Kiwi), and the biggest gains are on language pairs where the starting model is weak. Citation: *Reference-Free Reinforcement Learning Fine-Tuning for MT: A Seq2Seq Perspective*, [arXiv:2605.15976](https://arxiv.org/abs/2605.15976).

That is a different problem from ours. Our trainer always has `tgt`. GRPO would also multiply the cost: several full translations per sentence, then a reward model, on top of a normal training step. The IndicTrans2 authors and the WMT 2026 IndicTrans2 LoRA system did not need that once parallel text was available.

### DPO

DPO (Direct Preference Optimization) skips the reward model and learns straight from pairs: one chosen translation and one rejected translation. Citation: Rafailov et al., 2023, [arXiv:2305.18290](https://arxiv.org/abs/2305.18290).

The math has nothing to optimize if the dataset only contains the chosen side. This repo’s JSON, CSV, and the official IndicTrans2 folder layout are all one reference per source. Building fake rejected translations just to run DPO would throw away the simpler, proven SFT update.

### KTO

KTO (Kahneman-Tversky Optimization) learns from unpaired good/bad labels instead of chosen/rejected pairs. Citation: Ethayarajh et al., 2024, [arXiv:2402.01306](https://arxiv.org/abs/2402.01306).

A 2025 paper used KTO for Indian-language translation and beat an SFT baseline. Important detail: they fine-tuned **Llama 3.1 8B**, a general chat model, and they *manufactured* bad translations by progressively perturbing good ones, because parallel data was scarce. Citation: Bhaskar et al., 2025, *Progressive Perturbation with KTO for Enhanced Machine Translation of Indian Languages*, [MT Summit 2025](https://aclanthology.org/2025.mtsummit-1.26.pdf).

IndicTrans2 is already a translation model, trained on large parallel corpora. We are adapting it with real reference sentences. The KTO result does not say IndicTrans2 should abandon SFT. It says a general LLM, short on bitext, can improve if you invent negative samples. That is not this dataset.

## What would have to change before those methods made sense

They would belong in a **different project**, with different files:

- no reference translation at training time, only source sentences plus a quality score (the setting where GRPO-for-MT was studied), or
- a real preference file: two translations of the same source and a human judgment of which is better (DPO), or
- explicit good/bad labels collected on purpose (KTO).

Even then, the usual order is still: SFT on whatever parallel text you do have, and only afterwards a preference or reward stage. This repository stops at the first step, because that is the step our data supports and the step IndicTrans2’s own results are built on.

## What the code does

There is no `--train_method` flag. Training is supervised fine-tuning, and PPO, GRPO, DPO, and KTO are not implemented. The flag you do pass is `--train_mode` (`full`, `full_lora`, `lora`, or `qlora`), which only chooses which weights receive that update.
