# Fine-Tuning

Most of the following information is extracted from the [Fine-tuning](https://platform.openai.com/docs/guides/fine-tuning) section on the OpenAI website.
We strongly recommend checking the official site for the most up-to-date information.

## Advantages of fine-tuning  
Fine-tuning improves on few-shot learning by training on many more examples than can fit in the prompt
- Higher quality results than prompting
- Ability to train on more examples than can fit in a prompt
- Token savings due to shorter prompts
- Lower latency requests  

## When to fine-tune
#### Considerations before starting a fine-funing project
- Fine-tuning requires time and effort
- Recommend attempting the following first
  - Prompt engineering
  - Prompt chaining (breaking complex tasks into multiple prompts)
  - Function calling
#### Fine-tuning can improve results for "show, not tell"
- Setting the style, tone, format, or other qualitative aspects
- Improving reliability at producing a desired output
- Correcting failures to follow complex prompts
- Handling many edge cases in specific ways
- Performing a new skill or task that’s hard to articulate in a prompt
- Reducing costs and / or latency, by replacing GPT-4 or by utilizing shorter prompts 
#### Vs embeddings with retrieval
 - Embeddings with retrieval is best suited for large database of documents with relevant context and information
    - Make new information available by providing relevant context before generating its response
    - Retrieval strategies are not an alternative to fine-tuning and can in fact be complementary to it
- Fine-tuning can be used to make a model which is narrowly focused, and exhibits specific ingrained behavior patterns

## Fine-tuning project
#### Fine-tuning Jupyter notebook
We recommend checking the [Fine-tuning](https://platform.openai.com/docs/guides/fine-tuning) section on the OpenAI website for the most up-to-date information.
We prepare a [Jupyter Notebook](./fine-tuning_news.ipynb) that demonstrates the full cycle of a fine-tuning project.

#### Recommended practice
 - Experiment on instructions and prompts that work best for the model prior to fine-tuning
 - Required to provide at least 10 examples
 - Typically see clear improvements from fine-tuning on 50 to 100 training examples with gpt-3.5-turbo 
 - Recommend starting with 50 well-crafted demonstrations and seeing if the model shows signs of improvement after fine-tuning

#### Common commands
```
# Upload training file
openai.File.create(file=open("mydata.jsonl", "rb"),purpose='fine-tune')

# Start fine-tuning job
openai.FineTuningJob.create(training_file="file-abc123", model="gpt-3.5-turbo")

# List 10 fine-tuning jobs
openai.FineTuningJob.list(limit=10)

# Retrieve the state of a fine-tune
openai.FineTuningJob.retrieve("ftjob-abc123")

# Cancel a job
openai.FineTuningJob.cancel("ftjob-abc123")

# List up to 10 events from a fine-tuning job
openai.FineTuningJob.list_events(id="ftjob-abc123", limit=10)

# Delete a fine-tuned model (must be an owner of the org the model was created in)
openai.Model.delete("ft:gpt-3.5-turbo:acemeco:suffix:abc123")

# Use your fine-tuned model
completion = openai.ChatCompletion.create(
  model="ft:gpt-3.5-turbo:my-org:custom_suffix:id",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)
print(completion.choices[0].message)
```

#### More cookbooks from OpenAI
- https://cookbook.openai.com/examples/chat_finetuning_data_prep
- https://cookbook.openai.com/examples/how_to_finetune_chat_models
- https://cookbook.openai.com/examples/fine-tuned_classification
- https://cookbook.openai.com/examples/fine-tuned_qa/olympics-3-train-qa
- https://cookbook.openai.com/examples/fine-tuned_qa/olympics-2-create-qa
- https://cookbook.openai.com/examples/fine-tuned_qa/olympics-1-collect-data
