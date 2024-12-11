<div align="center">

# TRON Discord Bot

</div>

# The Community
**TRON** has one of the largest Discord communities I've ever seen. It is so big, that they have a completely separate discord community specifically for developers to get help. Blockchain technologies have always been **hard to access** for newbies, so we wanted to leverage the power of community and AI to make onboarding new talent easier. 

# The Project
We are a Discord bot that hooks into the [TRON developer documentation](https://developers.tron.network/), and uses that information to respond to user's questions automatically. By leveraging the newest models, we are able to better "stuff" the context window with as much TRON information as possible. 

# How to
We use [PDM](https://github.com/pdm-project/pdm)
to manage dependencies and execution. 

Before running the project, you need to create a `.env` file with the following content:
```
# Discord
discord_token="bot-token-here"

# LLMs for generating text
openai_api_key="sk-proj-xxxxxxxx"

# Storing embeddings
pinecone_api_key="xxxxx"
```

To run the project, run the following commands:
```bash
pdm install
pdm start
```

# The Tech
In general, we used:
* langchain
* discord.py
* openai
* pinecone

### Embeddings
We take a look at all the dev docs, and split it into "**useful chunks**". We take those chunks, and turn them into a vectors. We store these vectors into a vector database ([Pinecone](https://www.pinecone.io/), in this case), which allows us to query them later.

Now when a user sends a message, we can also turn that message into a vector. We can then pull all similar vectors from the vector database, and inject them into the prompt!

### Natural Language Processing (NLP)
Once we have all the vectors and their messages, we just have to construct a clever prompt that is really good at answering specific questions. This is just some basic prompt engineering, but then we inject all the embedding data so that we can learn about the newest TRON information live, then share that information with the world.

### Optimizations
Often, users will ask "bad questions" (e.g. questions that don't capture the full context of their problem). When this happens, the performance of embeddings tends to drop off. We introduce a new solution... We have a second agent to look at the user's question, and "hallucinate" user intention. We use these "guesses" as well as the original message to find our embeddings. 


