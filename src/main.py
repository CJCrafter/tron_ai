from typing import List

import discord
from discord import app_commands
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from src.ai.prompt import Prompt
from src.settings import SETTINGS


GUILD_ID = "1020927797931286561"


# Store a shared message history, so other users can join in on the conversation
message_history = []


class DiscordClient(discord.Client):
    def __init__(self, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return

        # Initialize the Pinecone index
        pinecone_api_key = SETTINGS.pinecone_api_key
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index("tron")

        # Initialize the OpenAI embeddings
        embeddings = OpenAIEmbeddings(api_key=SETTINGS.openai_api_key, model="text-embedding-3-large")

        user_question = message.content
        print(f"Received message: {user_question}")

        # Determine which embeddings we should be pulling from
        search_vectors: List[List[float]] = [embeddings.embed_query(user_question)]

        # Ask GPT to generate some more possible vectors
        from src.registers import REGISTERS
        embeddings_prompt = REGISTERS[Prompt]["find_embeddings"]
        embeddings_messages = [
            {"role": "system", "content": embeddings_prompt.build({})},
            {"role": "user", "content": user_question}
        ]
        guessed_embeddings = await embeddings_prompt.chat(embeddings_messages)
        for guess in guessed_embeddings.split("---"):
            print("Looking for vectors that match:", guess)
            search_vectors.append(embeddings.embed_query(guess))

        # Search the Pinecone index
        results: List[dict] = []
        for vector in search_vectors:
            print(f"Searching for vector: {vector[:2]}...")
            pinecone_vector = index.query(
                namespace=None,
                vector=vector,
                top_k=10,
                include_metadata=True
            )
            for match in pinecone_vector["matches"]:
                results.append(match)

        # Sort the results by cosine similarity
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        results_str = "\n".join([str(result["metadata"]["text"]) for result in results])[:2000]  # hard limit on number of chars
        print(results_str)

        # Get the prompt
        chat_prompt = REGISTERS[Prompt]["chat"]
        chat_messages = [
            *message_history,
            {"role": "user", "content": chat_prompt.build({"embeddings": results_str, "question": user_question})},
        ]
        response = await chat_prompt.chat(chat_messages)

        # Store a shared history between all users
        message_history.append({"role": "user", "content": user_question})
        message_history.append({"role": "assistant", "content": response})

        await message.reply(response)

    async def setup_hook(self) -> None:
        print("Syncing commands")
        try:
            synced = await self.tree.sync()
            print(f"Synced {len(synced)} command(s)")
            for command in synced:
                print(f"  - {command.name}")
        except Exception as e:
            print(f"An error occurred while syncing commands: {e}")


intents = discord.Intents.default()
intents.message_content = True

client = DiscordClient(intents=intents)

client.run(SETTINGS.discord_token)
