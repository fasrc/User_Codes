# OpenAI

In this document, you will find how to install and run a simple OpenAI example.

## What is OpenAI?

See OpenAI [website](https://openai.com/) and
[documentation](https://platform.openai.com/docs/introduction).

## Installation

You can install OpenAI in a conda/mamba environment:

```bash
[jharvard@boslogin01 ~]$ salloc --partition test --time 01:00:00 --mem-per-cpu 4G -c 2
[jharvard@holy8a24301 ~]$ module load python/3.10.12-fasrc01
[jharvard@holy8a24301 ~]$ export PYTHONNOUSERSITE=yes
[jharvard@holy8a24301 ~]$ mamba create -n openai_env openai
```

## Run OpenAI

You will need to provide an OpenAI key. You can generate one from
[https://platform.openai.com/api-keys](https://platform.openai.com/api-keys).

```bash
# Request an interactive job
[jharvard@boslogin01 ~]$ salloc --partition test --time 01:00:00 --mem-per-cpu 4G -c 2

# Source conda environment
[jharvard@holy8a24301 ~]$ mamba activate openai_env

# set OpenAI key
[jharvard@holy8a24301 ~]$ export OPENAI_API_KEY='my_key'

# set SSL_CERT_FILE with system's certificate
(openai_env) [jharvard@holy8a24301 ~]$ export SSL_CERT_FILE='/etc/pki/tls/certs/ca-bundle.crt'

# run OpenAI example
(openai_env) [jharvard@holy8a24301 ~]$ python openai-test.py
ChatCompletionMessage(content="In the realm of logic and code,\nWhere algorithms ebb and flow,\nThere lies a concept you must know,\nA dance called recursion, with a magical glow.\n\nWith a poem of loops and tangled rhyme,\nI'll unravel this tale, just give me time.\nImagine a tale within itself,\nA story that repeats, a tale that compels.\n\nAs the programmer sits, fingers poised,\nThey dream of a function that's ever poised,\nTo solve a problem with elegance and grace,\nUsing recursion's steps, in a gentle embrace.\n\nAt its heart, recursion starts with a call,\nA function that yields a problem small.\nA base case, the anchor, where it halts,\nBut beyond that point, the magic exalts.\n\nThrough loops and loops, it's a looping quest,\nUntil the base case, it finally rests.\nLike fractals spiraling into the infinite,\nRecursion unfolds, captivating and resolute.\n\nJust like a mirror placed before another,\nRecursion reflects, repeating in a smother.\nProblems break apart, into smaller fragments,\nSolving the pieces with recursive incense.\n\nWith each step, the path aims to amend,\nBreaking larger problems into commendable blend.\nLike a Russian doll, nested and profound,\nRecursion echoes its beauty, all around.\n\nIt's the Fibonacci sequence, climbing high,\nIt's the maze solver, searching the sky.\nA symphony of puzzles, all intertwined,\nRecursion guides, with a creative mind.\n\nBut beware, dear coder, of the infinite call,\nFor without careful restraint, it will befall.\nA stack overflow, a memory's demise,\nRecursion has limits, heed them and be wise.\n\nSo, wrap your mind 'round this tale of mine,\nLet recursion's magic forever shine.\nIn the realm of programming, it's a tool profound,\nA dance of complexity, where solutions are found.", role='assistant', function_call=None, tool_calls=None)
```

**Note:** OpenAI uses the python package `httpx`. You must set the variable
`SSL_CERT_FILE` to use the system's certificate. If you do not set
`SSL_CERT_FILE` OpenAI will give this error:

```bash
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1006)
```

