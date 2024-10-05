from pathlib import Path

import discord
from discord import app_commands

from src.settings import SETTINGS


GUILD_ID = "1020927797931286561"


class DiscordClient(discord.Client):
    def __init__(self, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return

        await self.process_commands(message)

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
