# Generative Pre-Trained Transformer (GPT)

:::{admonition} OpenAI's GPT-series are Large Language Models
:class: tip
Let's break down these terms in reverse order:

- `Language Models`
- `Large`
- `Transformer`
- `Pre-Trained`
- `Generative`
:::

:::{admonition} **[Language Model](https://en.wikipedia.org/wiki/Language_model)**
:class: tip
A `Language Model` (LM) is a model that assigns probabilities to sequences of words. Probabilities are estimated (the models are trained) using collections of naturally ocurring text, called a corpus. You can use a LM to select the most probable next word given a context of a preceding series of words. 
:::

:::{card} [Animated Transfomer](https://prvnsmpth.github.io/animated-transformer/)
<video width="99%" height="540" autoplay loop muted markdown="1">
    <source src="_images/animated-transfomer.mp4" type="video/mp4" markdown="1" >
</video>
:::


```{figure} ./images/lm-hist.png
---
width: 800px
name: lm-hist
---
[Evolution of language models over time](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch01.html#understanding_the_transformer_architecture_and_its)
```

:::{admonition} **[Large](https://en.wikipedia.org/wiki/Large_language_model#List)**
:class: tip
We call a language model `large` when:
- The model has billions of parameters (approaching a trillion in frontier models)
- The model has been trained on billions or trillions of words/tokens

Why the focus on large language models (LLMs)? Because as scale increases new `emergent` capabilities have appeared, such as complex reasoning. Smaller language models gave no indication that this would happen.
:::

```{figure} ./images/wikipedia-list.png
---
width: 600px
name: wikipedia-list
---
[LLM List](https://en.wikipedia.org/wiki/Large_language_model#List)
```

:::{admonition} **Transformer**
:class: tip
`Transformers` were the key innovation that allowed language models to get large. They are a deep learning architecture that allow massive parallelization of training and inference on GPUs
:::

```{figure} ./images/The-Transformer-model-architecture.png
---
width: 600px
name: trans-subset
---
```

```{figure} ./images/attention-is-all-you-need.png
---
width: 600px
name: attention
---
[Attention Paper](https://arxiv.org/abs/1706.03762)
[Annotated Version](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
```

:::{admonition} **Pre-trained**
:class: tip
`Pre-trained` language models have been trained via self-supervision on vast quantities of text. In our current context of LLMs, these are also sometimes called [`foundation`](https://en.wikipedia.org/wiki/Foundation_models) models. Pre-trained models can be used as is, or trained further, for exmaple with "fine-tuning" in order to improve performance on specific target domains.
:::

:::{admonition} **Generative**
:class: tip
`Generative` models are models that create content as output, in this case linguistic content. Vanilla pre-trained generative models do simple next word prediction (OpenAI calls these "GPT base"). They predict the most likely next word in a sequence. These can be useful for some tasks, such as speech recognition and predictive text completion. The assistant-style models we are more accustomed to using are models that have been further trained via supervised fine-tuning and reinforcement learning from human feedback (RLHF) to behave in a useful and safe manner, for example by responding to questions with answers like an assistant.
:::

