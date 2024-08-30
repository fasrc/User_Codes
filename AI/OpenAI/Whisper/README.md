# OpenAI Whisper

In this document, you will find how to install and run a simple Whisper example.

## What is Whisper?

See Whisper [website](https://openai.com/index/whisper/) and
[documentation](https://platform.openai.com/docs/guides/speech-to-text).

## Security

Please, **carefully** read Harvard's [AI
guidelines](https://huit.harvard.edu/ai/guidelines) and [Generative AI
tool comparision](https://huit.harvard.edu/ai/tools).

You can only use openAI and other genAI on non-sensitive (data
security level 1) public data on Cannon. See [FASRC
Guidelines](https://docs.rc.fas.harvard.edu/kb/openai/).

For data security ***levels 2 and 3***, you need to work with your
school to discuss your needs. It is your responsibility to make sure
you get setup with the appropriate contractual coverage and
environment -- esp. to avoid having the model learn from your input
and leak sensitive information.

## Installation

You can install Whisper in a conda/mamba environment. Assuming that
you have already created and installed OpenAI as described
[here](https://github.com/fasrc/User_Codes/tree/master/AI/OpenAI),
Whisper can be easily installed as mentioned below:

```bash
[jharvard@boslogin01 ~]$ salloc --partition=gpu_test --gres=gpu:1 --time=01:00:00 --mem-per-cpu=4G --cpus-per-task=2
[jharvard@holy8a24301 ~]$ module load python
[jharvard@holy8a24301 ~]$ export PYTHONNOUSERSITE=yes
[jharvard@holy8a24301 ~]$ source activate openai_env
[jharvard@holy8a24301 ~]$ pip install -U openai-whisper
[jharvard@holy8a24301 ~]$ mamba install conda-forge::ffmpeg -y
[jharvard@holy8a24301 ~]$ mamba install conda-forge::rust -y
[jharvard@holy8a24301 ~]$ conda deactivate
```

## Run OpenAI

As mentioned
[here](https://github.com/fasrc/User_Codes/tree/master/AI/OpenAI), you
will need to provide an OpenAI key. If you haven't already, you can
generate one from
[https://platform.openai.com/api-keys](https://platform.openai.com/api-keys). It
is advisable to store this key in a safe location, like your
`~/.bashrc` file on Cannon. Additionally, you can also store the
`SSL_CERT_FILE` in your `~/.bashrc` instead of exporting it every time
you want to run OpenAI or Whisper.


> **_NOTE:_** You would need to download the sample audio file, 'harvard.wav', and the
example script, `openai-whisper-test.py`, to your profile on Cannon in
order to run this example. Remember to update the path of `harvard.wav`
in `openai-whisper-test.py` to its location on Cannon for the example to
work properly.


```bash
# Request an interactive job
[jharvard@boslogin01 ~]$ salloc --partition test --time 01:00:00 --mem-per-cpu 4G -c 2

# Source conda environment
[jharvard@holy8a24301 ~]$ mamba activate openai_env

# Replace my_key with the key that you generated on OpenAI's website
[jharvard@holy8a24301 ~]$ export OPENAI_API_KEY='my_key'

# Set SSL_CERT_FILE with system's certificate
(openai_env) [jharvard@holy8a24301 ~]$ export SSL_CERT_FILE='/etc/pki/tls/certs/ca-bundle.crt'

# Run Whisper example
[//]: # "You would need to download the sample audio file, 'harvard.wav', and the
example script, `openai-whisper-test.py`, to your profile on Cannon in
order to run this example. Remember to update the path of `harvard.wav`
in `openai-whisper-test.py` to its location on Cannon for the example to
work properly."

(openai_env) [jharvard@holy8a24301 ~]$ python openai-whisper-test.py
The stale smell of old beer lingers. It takes heat to bring out the
odor. A cold dip restores health and zest. A salt pickle tastes fine
with ham. Tacos al pastor are my favorite. A zestful food is the hot
cross bun.
```
