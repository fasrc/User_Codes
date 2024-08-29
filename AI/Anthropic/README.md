# Anthropic Claude

In this document, you will find how to install and run a simple Claude example.

## What is Claude?

See Claude [website](https://www.anthropic.com/claude) and
[documentation](https://docs.anthropic.com/en/docs/intro-to-claude).

## Security

Please, **carefully** read Harvard's [AI guidelines](https://huit.harvard.edu/ai/guidelines) and [Generative AI tool comparision](https://huit.harvard.edu/ai/tools).

You can only use Anthropic tools and other genAI models on
non-sensitive data (data security level 1) public data on Cannon.

For data security ***levels 2 and 3***, you need to work with your
school to discuss your needs. It is your responsibility to make sure
you get setup with the appropriate contractual coverage and
environment -- esp. to avoid having the model learn from your input
and leak sensitive information.

## Installation

You can install Claude in a conda/mamba environment:

```bash
[jharvard@boslogin01 ~]$ salloc --partition=test --time=02:00:00 --mem=8G --cpus-per-task=2
[jharvard@holy8a24301 ~]$ module load python
[jharvard@holy8a24301 ~]$ export PYTHONNOUSERSITE=yes
[jharvard@holy8a24301 ~]$ mamba create --name claude_env python -y
[jharvard@holy8a24301 ~]$ source activate claude_env
[jharvard@holy8a24301 ~]$ pip install anthropic
[jharvard@holy8a24301 ~]$ conda deactivate
```
## Run Claude

You will need to provide an Anthropic API key. You can generate one
from their [API
page](https://console.anthropic.com/login?selectAccount=true&returnTo=%2Fsettings%2Fkeys%3F). Also,
see their [quickstart
guide](https://docs.anthropic.com/en/docs/quickstart).

```bash
# Request an interactive job
[jharvard@boslogin01 ~]$ salloc --partition=gpu_test --gres=gpu:1 --time=02:00:00 --mem=8G --cpus-per-task=2

# Source conda environment
[jharvard@holy8a24301 ~]$ mamba activate claude_env

# replace my_key with the key that you generated on the Anthropic API website
(claude_env) [jharvard@holy8a24301 ~]$ export ANTHROPIC_API_KEY='Your-API-Key'

# set SSL_CERT_FILE with system's certificate
(claude_env) [jharvard@holy8a24301 ~]$ export SSL_CERT_FILE='/etc/pki/tls/certs/ca-bundle.crt'

# run Claude example
(claude_env) [jharvard@holy8a24301 ~]$ python claude_quickstart.py
```

**Note:** Anthropic uses the python package `httpx`. You must set the variable
`SSL_CERT_FILE` to use the system's certificate. If you do not set
`SSL_CERT_FILE` Anthropic will give this error:

```bash
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1006)
```
