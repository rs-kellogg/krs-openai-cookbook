# Takeaways

:::{admonition} Summary
:class: note
- Setting up an OpenAI account
- Multimodal OpenAI use cases
- Advanced topic: Function calling and RAG
:::

:::{card} Development Workflow
- Identify a dataset and specify precisely what you want to do with it
- Engineer a prompt (use OpenAI Playground)
- Evaluate performance on a sample
    * If performance is unacceptable, try further prompt engineering, functions, etc.
    * If performance is still not good enough, try fine-tuning
- Deploy at scale

```{figure} ./images/gpt-dev-cycle.png
---
width: 400px
name: gpt-dev-cycle
---
```
:::

:::{card} Structuring and Testing Code
- Structure your code into source code (functions) along with unit tests
- Functions are composed in scripts, which should have runtime tests and generate logs
- All code (source, scripts, tests) need to be in source control, typically `git`
- Save logs and output files in a secure location. Do not modify them.
```{figure} ./images/testing.png
---
width: 400px
name: testing
---
```
:::

:::{admonition} Moving forward
:class: important
- This is a living document, we will be updating it with new topics regularly
- We will also be adding code to the attached `openai-helper` package
- Come to us for help during office hours at 5424, or send us an email us at <url><p><a href="mailto:rs@kellogg.northwestern.edu">rs@kellogg.northwestern.edu</a></p>
:::