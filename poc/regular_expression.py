import re


def patterns(text):
    text = re.sub(r'#[A-Za-z0-9_]{1,40}', '[HASTAG]', text)
    text = re.sub(r'@[A-Za-z0-9_]{1,40}', '[MENTION]', text)
    return text

text = "Why not try something new today? Explore the #PanamaPapers data set with a #graphdatabase sandbox. " \
       "Easy to get started with, friendly step-by-step guide, all free and no download required! #Neo4j #learning @fulanito"

print(patterns(text))

result = text.split(' ')
len(result)
