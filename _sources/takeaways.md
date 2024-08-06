# Takeaways

:::{admonition} Summary
:class: note
- Setting up and using your OpenAI account
- Developing a testable, reprocucible workflow that scales
:::

:::{admonition} Development Workflow
:class: note
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

:::{admonition} Structuring and Testing Code
:class: note
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
- This is a living document, we will be updating it regularly
- We will also be adding code to the attached `openai-helper` package
:::