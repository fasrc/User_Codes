# GenAI-Tools Workshop -- Extras

## Generate an API key for an OpenAI services
Generate API key for OpenAI in the [Harvard API Portal](https://portal.apis.huit.harvard.edu/)
* See [Register an App](https://portal.apis.huit.harvard.edu/register-an-app) for how to create the API key
  * (FAS/HUIT) Free ($10/month limit) [OpenAI for Community Developers](https://portal.apis.huit.harvard.edu/docs/ais-openai-direct-limited/1/overview) (*must be signed in to follow the link*)

## Codex CLI

### Installation

On the FASRC cluster, use mamba to install:
```
module load python
mamba create -n codex codex
mamba activate codex
```

### Setup (ChatGPT EDU)

If you have a [OpenAI ChatGPT EDU](https://www.huit.harvard.edu/openai-chatgpt-edu) account:

    codex login --device-auth

Follow directions, authenticating with your ChatGPT EDU account.

*Note: if you have already authenticated to ChatGPT EDU using the VS Code Codex extension on the FASRC cluster, this step may be skipped, as authentication state is shared between the codex CLI and VS Code extension*

Follow 

1. Generate an OpenAI-compatible API key in the Harvard API portal (see above)

2. Add environment variable in your ~/.profile (or ~/.bash_profile) file:

    export HARVARD_OPENAI_API_KEY=...API Key...

3. Source ~/.profile (~/.bash_profile) (or start a new login shell)

    source ~/.profile

4. Create ~/.codex directory (if it doesn't already exist)

    mkdir -p ~/.codex

5. Add the following to ~/.codex/config.toml:

```
model_provider = "harvard_openai"
model = "gpt-5.3-codex"

[model_providers.harvard_openai]
name = "Harvard OpenAI Gateway"
base_url = "https://go.apis.huit.harvard.edu/ais-openai-direct-limited-schools/v1"
wire_api = "responses"
env_http_headers = { "api-key" = "HARVARD_OPENAI_API_KEY" }
```

### Start codex

To start codex:

```
cd ...directory with your repository/project ...
codex
```

* By default, codex is [sandboxed](https://developers.openai.com/codex/concepts/sandboxing) to the directory the command is executed from
  - Only subdirectories accessible without explicitly granting permission

## Claude Code for VS Code

1. Obtain [HUIT AI Services - AWS Bedrock](https://portal.apis.huit.harvard.edu/docs/ais-bedrock-llm/2/overview) API key

2. In VS Code, install [Claude Code for VS Code](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code) extension
  - See [VSCode Remote Development via SSH and Tunnel](https://docs.rc.fas.harvard.edu/kb/vscode-remote-development-via-ssh-or-tunnel/) for instructiosn on launching VS Code on the FASRC cluster


3. Configure Claude Code for VS Code to use the HUIT endpoint.
  Update the following settings **in the remote**:
  - check "Claude Code: Disable Login Prompt"
  - Under Claude Code: Environment Variables, click "Edit in settings.json", and fill in:

  ```
  "claudeCode.environmentVariables": [
    {"name": "ANTHROPIC_BEDROCK_BASE_URL",
     "value": "https://apis.huit.harvard.edu/ais-bedrock-llm/v2"},
    {"name": "ANTHROPIC_API_KEY",
     "value": "...Your API Key..."},
    {"name": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
     "value": "us.anthropic.claude-haiku-4-5-20251001-v1:0"},
    {"name": "CLAUDE_CODE_SKIP_BEDROCK_AUTH",
     "value": "1"},
    {"name": "CLAUDE_CODE_USE_BEDROCK",
     "value": "1"}
  ]
  ```
