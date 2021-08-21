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

bot = commands.Bot("!", intents=discord.Intents.default())

bot.messageCache = {}
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}!')

@bot.event
async def on_message(message):
    bot.messageCache[str(message.author.id)] = bot.messageCache.get(str(message.author.id), " ") + message.content
    # if not message.author.bot and message.content.startswith(":sb"):
    #     commandElements = list(map(str, message.content.strip(":sb").split()))
    #     if len(commandElements) >= 2:
    #         commandName = commandElements[0]
    #         commandArgs = commandElements[1:]
    #     elif len(commandElements) == 1:
    #         commandName = commandElements[0]
    await bot.process_commands(message)


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



      


bot.run("ODc4NDcxOTk3NTg2MjkyNzc4.YSBqzQ.NUSHwH5J1-b_tPBNPJSzQ_HkJPE")
