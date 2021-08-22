import atexit

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
from discord.ext import commands

import identifier

bot = commands.Bot("!", intents=discord.Intents.default())

#this takes the info from the jsons which are for offline storage
bot.messageCache = {}
bot.spammerList = {}
try:
    with open("messages.json", "r") as messageFile:
        bot.messageCache = json.loads(messageFile.read())
    with open("spamlist.json", "r") as spamFile:
        bot.spammerList = json.loads(spamFile.read())
except:
    bot.messageCache = {}
    bot.spammerList = {}

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}!')

@bot.event
async def on_message(message):
    bot.messageCache[str(message.author.id)] = bot.messageCache.get(str(message.author.id), " ") + message.content

    await bot.process_commands(message)

@bot.command(name="logspam")
async def _logspam(ctx):
    if not ctx.author.guild_permissions.administrator:
        return await ctx.send(f"Sorry, logging of spambots is for admins only! {bot.get_emoji(691757044361068574)}")
    else:
        userList = ctx.message.mentions
        for user in userList:
            bot.spammerList[str(user.id)] = 1

    print(bot.spammerList)

@bot.command()
async def ping(ctx):
    await ctx.send("Pong!")

@bot.command(name="eval", aliases=["ev"])
async def _eval(ctx, *, code: str):
    if not dev_check(ctx.author.id):
        return await ctx.send(f"Sorry, but you can't run this command because you ain't a developer! {bot.get_emoji(691757044361068574)}")
    env = {
        "bot": bot,
        "ctx": ctx,
        "channel": ctx.channel,
        "author": ctx.author,
        "guild": ctx.guild,
        "message": ctx.message,
        "msg": ctx.message,
        "_": bot._last_result,
        "source": inspect.getsource,
        "src": inspect.getsource,
        "session": bot.session,
        "docs": lambda x: print(x.__doc__)
    }

    env.update(globals())
    body = cleanup_code(code)
    stdout = io.StringIO()
    err = out = None
    to_compile = f"async def func():\n{textwrap.indent(body, '  ')}"
    stopwatch = Stopwatch().start()

    try:
        exec(to_compile, env)
    except Exception as e:
        stopwatch.stop()
        err = await ctx.send(f"**Error**```py\n{e.__class__.__name__}: {e}\n```\n**Type**```ts\n{Type(e)}```\n⏱ {stopwatch}")
        return await ctx.message.add_reaction(bot.get_emoji(522530579627900938))

    func = env["func"]
    stopwatch.restart()
    try:
        with redirect_stdout(stdout):
            ret = await func()
            stopwatch.stop()
    except Exception as e:
        stopwatch.stop()
        value = stdout.getvalue()
        err = await ctx.send(f"**Error**```py\n{value}{traceback.format_exc()}\n```\n**Type**```ts\n{Type(err)}```\n⏱ {stopwatch}")
    else:
        value = stdout.getvalue()
        if ret is None:
            if value:
                try:
                    out = await ctx.send(f"**Output**```py\n{value}```\n⏱ {stopwatch}")
                except:
                    paginated_text = paginate(value)
                    for page in paginated_text:
                        if page == paginated_text[-1]:
                            out = await ctx.send(f"```py\n{page}\n```", edit=False)
                            break
                        await ctx.send(f"```py\n{page}\n```", edit=False)
                    await ctx.send(f"⏱ {stopwatch}", edit=False)
        else:
            bot._last_result = ret
            try:
                out = await ctx.send(f"**Output**```py\n{value}{ret}```\n**Type**```ts\n{Type(ret)}```\n⏱ {stopwatch}")
            except:
                paginated_text = paginate(f"{value}{ret}")
                for page in paginated_text:
                    if page == paginated_text[-1]:
                        out = await ctx.send(f"```py\n{page}```", edit=False)
                        break
                    await ctx.send(f"```py\n{page}```", edit=False)
                await ctx.send(f"**Type**```ts\n{Type(ret)}```\n⏱ {stopwatch}", edit=False)
        if out:
            await ctx.message.add_reaction(bot.get_emoji(522530578860605442))
        elif err:
            await ctx.message.add_reaction(bot.get_emoji(522530579627900938))
        else:
            await ctx.message.add_reaction("\u2705")

# updates the JSON files with the new info
@atexit.register
def before_exit():
    with open("messages.json", "w") as messageFile:
        #messageFile.truncate(0)  # clears file
        messageFile.write(json.dumps(bot.messageCache))
    with open("spamlist.json", "w") as spamFile:
        #spamFile.truncate(0)  # clears file
        spamFile.write(json.dumps(bot.spammerList))

    print("here")


#-----------------MAIN------------------
bot.run("ODc4NDcxOTk3NTg2MjkyNzc4.YSBqzQ.NUSHwH5J1-b_tPBNPJSzQ_HkJPE")

