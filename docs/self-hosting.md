# Self-hosting OpenBapo

This guide sets up your own OpenBapo bot: a Discord application, a machine with a GPU to run it, and the settings that control it. Expect about half an hour, plus the time it takes to download roughly 9 GB of model weights.

- [1. Create the Discord application](#1-create-the-discord-application)
- [2. Install and run](#2-install-and-run)
- [3. Settings](#3-settings)
- [4. What is stored, and for how long](#4-what-is-stored-and-for-how-long)
- [5. Troubleshooting](#5-troubleshooting)

## What you need

| | |
|---|---|
| **Machine** | Linux, Python 3.11 or newer |
| **GPU** | NVIDIA, 12 GB of memory or more recommended (tested on an RTX 3090, 24 GB). A 4-minute song takes about 70 seconds. A CPU works but is many times slower. |
| **Disk** | About 15 GB: ~9 GB of model weights plus PyTorch |
| **System packages** | `ffmpeg`, `libcairo2` and a CJK font (`fonts-noto-cjk`) |
| **Network** | Access to Discord, Hugging Face and download.pytorch.org (for the first download only) |

Run **one** instance per bot token. The bot keeps its queue in a SQLite file, which belongs on a local disk, not a network share.

## 1. Create the Discord application

1. Open the [Discord Developer Portal](https://discord.com/developers/applications) and click **New Application**. Give it a name and, if you like, an icon.
2. **Bot** page: click **Reset Token** and copy the token. It is a password for your bot. Keep it out of chats, screenshots and Git. No privileged gateway intents are needed; leave Presence, Server Members and Message Content off.
3. **Installation** page: tick both **Guild Install** and **User Install**.
   - Guild Install: scopes `applications.commands` and `bot`, with these permissions:
     **View Channels**, **Send Messages**, **Embed Links**, **Attach Files**, **Read Message History**, **Create Public Threads** and **Send Messages in Threads**.
   - User Install: scope `applications.commands`.
4. Leave **Interactions Endpoint URL** empty. The bot receives everything over its gateway connection and opens no ports.

The install link for a server, with your application's ID from the **General Information** page:

```
https://discord.com/oauth2/authorize?client_id=YOUR_APPLICATION_ID&scope=bot+applications.commands&permissions=309237763072
```

The bot registers its commands (`/transcribe`, `/status`, `/delete`, `/help` and the **Transcribe** message command) itself when it starts. New commands can take a minute to appear in Discord; restarting your Discord client (Ctrl+R) helps.

### Suggested server setup

- A channel such as `#transcribe` where people use the bot. Each song gets its own thread there, so the channel shows one line per song.
- Other channels without the bot. In **Server Settings → Integrations → OpenBapo** you can choose which channels its commands work in. Server owners and administrators can always use them everywhere.
- People who want a song kept private can send `/transcribe` to the bot in a direct message.

## 2. Install and run

### From source

```bash
sudo apt install ffmpeg libcairo2 fonts-noto-cjk git

git clone https://github.com/Ariza-Sun/OpenBapo.git
cd OpenBapo
python -m venv .venv && source .venv/bin/activate

# PyTorch first, built for your CUDA version (https://pytorch.org/get-started/locally/).
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# OpenBapo with the bot, rendering and the model packages, at tested versions.
python deploy/install.py

openbapo download-models

export DISCORD_TOKEN=...                       # better: load it from a file only you can read
export OPENBAPO_DATA_DIR=/srv/openbapo/data    # where the queue and the sheets live
openbapo-bot
```

`deploy/install.py` installs the model packages pinned in `deploy/constraints.txt`. One of them, lv-chordia, asks for a newer PyTorch than the one tested here, so it is installed without its dependencies; `pip check` will mention this and it is expected.

Keep the bot running with your usual tool: a systemd service, `tmux`, or Docker below.

### With Docker

You need Docker with Compose and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
cp .env.example .env
chmod 600 .env
# Put your token in .env with an editor: DISCORD_TOKEN=...
docker compose build
docker compose run --rm bot openbapo download-models
docker compose up -d
docker compose logs --tail 50 bot
```

Two volumes keep data across restarts: `results` (the queue and the sheets) and `models` (the weights). The container runs as an unprivileged user, has no open ports, and reports itself unhealthy if the bot stops reaching Discord.

> [!NOTE]
> The Docker setup has not yet been through a test on a clean machine. If something in it does not work for you, please open an issue.

## 3. Settings

All settings are environment variables. Only `DISCORD_TOKEN` is required.

| Variable | Default | What it does |
|---|---|---|
| `DISCORD_TOKEN` | — | The bot token from the Developer Portal |
| `OPENBAPO_DATA_DIR` | `./var/openbapo` | Where the queue database and the work files live |
| `OPENBAPO_DEVICE` | `cuda` | `cuda`, `cpu` or `auto` |
| `OPENBAPO_RECOGNIZER` | `qwen3-asr` | Lyrics recognizer. `whisper` uses less memory; download it with `openbapo download-models --recognizer whisper` |
| `OPENBAPO_MAX_UPLOAD_BYTES` | `20971520` (20 MB) | Largest file accepted. Discord's own limit for free accounts is 10 MB |
| `OPENBAPO_CONTACT` | — | An email or web address shown in `/help` and on errors, so users can reach you |
| `OPENBAPO_UNLIMITED_USERS` | — | Comma-separated Discord user IDs without a daily limit, usually your own |
| `DISCORD_PROXY` | — | An HTTP proxy for all Discord traffic, for machines that cannot reach Discord directly |
| `HF_HOME`, `TORCH_HOME` | the usual caches | Where model weights are stored |

Built-in limits: 5 songs per person per day (resets at midnight UTC), songs up to 10 minutes, at most 20 songs waiting, and 15 minutes from upload to sheet before a job is stopped.

## 4. What is stored, and for how long

- **Audio** is saved to the work directory only while a song waits and is analyzed, then deleted, also when a job fails, is cancelled or times out. The separated stems never leave memory.
- **Sheets** (the analysis and the user's chosen options) are kept for **30 days** so users can come back to them with `/status`, then deleted.
- **`/delete`** stops a user's running job and deletes all their sheets at once. It also removes the bot's messages and threads for those songs where Discord allows it.
- **Daily quotas** are counted under a keyed hash of the user's ID, not the ID itself, and forgotten the next day.
- **Logs** contain job IDs, timings and error types. They never contain tokens, file contents, lyrics or error messages from the models.
- Files a user posts to Discord stay on Discord; OpenBapo cannot delete them.

Nothing is used for training.

## 5. Troubleshooting

**The commands don't show up in Discord.** Wait a minute and restart the Discord client. Check that the bot log says `connected to Gateway`.

**The bot can't connect.** Some networks block Discord. Set `DISCORD_PROXY=http://host:port` to an HTTP proxy that can reach it.

**Songs fail with "the server's GPU is busy".** Another program is using the GPU's memory. Free it, give the bot its own GPU with `CUDA_VISIBLE_DEVICES`, or switch to `OPENBAPO_RECOGNIZER=whisper`, which needs less.

**Songs fail with a service error.** The bot's log has a line like `job=… failed error=service where=ValueError at pipeline.py:131`. Please include that line when you open an issue.

**No thread is opened for a song.** The bot needs **Create Public Threads** and **Send Messages in Threads** in that channel. Without them it posts the sheet in the channel instead.

**Users can't upload files over 10 MB.** That limit is Discord's, for accounts without Nitro and servers without boosts.
