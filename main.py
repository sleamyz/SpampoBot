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
from utils.stopwatch import Stopwatch
from utils.type import Type
from contextlib import redirect_stdout
from discord.ext import commands
import json
import ezjson
import colorama
import difflib
import math
from box import Box
from motor.motor_asyncio import AsyncIOMotorClient
from ext.context import DatContext
from ext.logger import Logger as logger
from cogs.utils.utils import Utils
from audio.AudioManager import AudioManager

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
