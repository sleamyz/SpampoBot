import discord
import re
import os
import io
import traceback
import sys
import time
import datetime
import asyncio
import random
import aiohttp
import random
import textwrap
import inspect
import json

client = discord.Client()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if not message.author.bot and message.content.startswith(":sb"):
        commandElements = list(map(str, message.content.strip(":sb").split()))
        if len(commandElements) >= 2:
            commandName = commandElements[0]
            commandArgs = commandElements[1:]
        elif len(commandElements) == 1:
            commandName = commandElements[0]




      


client.run()
