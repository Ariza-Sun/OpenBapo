# 自部署 Discord Bot（第 6 步）

当前已完成本地离线交互测试和真实 GPU 任务验证。**真实 Discord 服务器 / 用户安装私信验收、镜像构建、干净机器 30 分钟部署验收尚未完成**。以下是待实机核验的部署步骤，不能据此宣称已上线。

## 准备环境

- Linux、Docker Engine、Compose、NVIDIA 驱动与 NVIDIA Container Toolkit。
- 一张 GPU；本地验证使用 RTX 3090 24 GB。CPU 能跑但未承诺满足 5 分钟目标。
- 为模型缓存和镜像预留足够磁盘空间；首次需要联网下载多份大权重。首次下载不计入暖缓存性能。
- Discord Developer Portal 中建立应用，获取 Bot Token，在部署机配置，**不要发到聊天里或提交仓库**。

Compose 的 GPU 配置沿用 [Docker 官方方式](https://docs.docker.com/compose/how-tos/gpu-support/)；只启动一个服务实例，SQLite 文件放本机卷，不放网络文件系统。不要增加副本数。

## 配置 Discord

1. 在应用 Installation 中启用 **Guild Install** 和 **User Install**。
2. Guild Install 的 scopes 选择 `applications.commands`、`bot`；授予 View Channels、Send Messages、Embed Links、Attach Files、Read Message History（线程内使用时还需 Send Messages in Threads）。
3. User Install 使用 `applications.commands`；允许用户将应用安装到自己的账号。
4. 无需开启 Message Content、Members 或 Presence 特权 intent。消息右键命令从交互载荷取得目标附件。
5. 本应用通过 Gateway 接收交互，**不填写 Interactions Endpoint URL**。命令启动时全局同步，支持服务器、Bot 私信及用户安装的私聊上下文；不要用只同步测试 guild 的方式代替，否则无法测用户安装。

参见 [Discord 用户安装指南](https://docs.discord.com/developers/tutorials/developing-a-user-installable-app)。服务内命令会注册中文本地化名称；英文命令为 `/transcribe`、`/status`、`/delete`、`/help`，消息右键为 `Transcribe`。

## 构建并启动

在项目根目录：

```bash
cp .env.example .env
chmod 600 .env
# 用编辑器在 .env 中填写 DISCORD_TOKEN，不在命令行或日志中打印。
docker compose config --quiet
docker compose build
docker compose run --rm bot python scripts/warm_service_models.py
docker compose up -d
docker compose logs --tail 50 bot
```

预热命令依次下载 / 加载分析模型，默认在 CPU 完成，不接受用户任务。初次联网下载可能超过 30 分钟，必须在实际部署验收中分别记录下载与启动耗时。镜像不内置权重。

数据卷 `results` 保存 SQLite 和尚在处理的临时文件；`models` 保存模型缓存。Bot 以非 root 用户运行，不开放 HTTP 端口。健康检查观察 Gateway 就绪后的服务心跳；Docker 的 health 状态本身不会自动重启 unhealthy 容器，应同时查看日志。服务进程退出则按 Compose 重启策略重启。

`deploy/constraints.txt` 固定本次验证使用的主要模型依赖版本。`lv-chordia` 的固定提交声明 `torch>=2.13`，但项目既有模型回归和本次队列实测使用 `torch 2.5.1+cu121`。`deploy/install.py` 单独以 `--no-deps` 安装它，并显式补齐其它依赖；这是已知兼容性例外，`pip check` 会报告该版本声明不满足。更新 torch 或模型依赖后必须重新验证，不能只依据安装成功。

如不使用容器，可复用已有推理虚拟环境：

```bash
pip install -e '.[bot,render]'
export OPENBAPO_DATA_DIR=/absolute/path/to/private-service-data
export OPENBAPO_DEVICE=cuda
# 通过部署环境安全设置 DISCORD_TOKEN。
openbapo-bot
```

推理依赖、ffmpeg、libcairo2 与中文字体仍需预先安装；CTranslate2 需要匹配的 cuDNN / cuBLAS 动态库路径，容器入口脚本已自动配置。

## 用户流程与数据

- `/transcribe audio:<附件> [title] [artist] [lyrics]`。六种音频格式、8 分钟上限；服务默认 10 MiB，可设置 `OPENBAPO_MAX_UPLOAD_BYTES`，同时仍受 Discord 限制。
- 先立即私有 defer，完成格式 / 时长校验后创建公开排队回执，成功时把这条回执更新为谱子。无需刷新频道，控件只改同一消息。
- 每人每 UTC 日 5 首；无效输入不计数，已入队任务（包括后续失败 / 取消）计数。队列上限 20；预计时间为估计值。入队后连同排队最多 15 分钟。
- 自动识别可能漏掉短而孤立的人声；上传时附歌词走独立对齐路径。没有歌词仍出和弦谱，旋律简谱行等待第 7 步。
- `/status` 列出最近 10 个任务，`/status job:<完整编号>` 取回指定结果。原消息交互凭据过期时尝试原频道，再尝试私信；两者失败会保留结果供取回。
- 控件只允许发起者使用。调整和导出只读已保存的分析结果，不重跑模型；PNG 多页打包为 ZIP，PDF 多页合并，MusicXML 可继续精修。
- `/delete` 取消并等待正在运行的子进程退出，再删除该用户所有服务端任务。尝试撤回最新记录的 Bot 谱子消息；用户安装且旧凭据已过期时，Bot 可能无权删除历史消息，会明确提示。用户自己上传的 Discord 附件与 Discord 的存储由用户另行管理。
- 原音频、临时分轨和上传的歌词请求文件在任务结束 / 取消 / 超时后清理。结果和参数保留 30 天，每分钟检查过期；服务器停机期间不会运行清理，重启后补做。交互 token 在投递完成或失效后删除。
- `/delete` 不重置当日配额。服务仅保留当日 HMAC 标识及次数，第二天清理，不在该表保存 Discord 用户 ID。SQLite 使用安全删除和 WAL 截断；不承诺底层磁盘、宿主机备份或 Discord CDN 的物理擦除。
- 不保存音频衍生物或把用户数据用于训练。日志只含任务 ID、耗时和错误类型，不写 token、原歌词或模型异常原文。完整阶段耗时保存在任务 `timings` 字段。

## 验证

不连接 Discord 的测试与计时：

```bash
pytest tests/test_service.py tests/test_bot.py
python scripts/smoke_service.py /path/to/4-minute-song.mp3 --device cuda --out /tmp/service-smoke
```

本地实测见 [第 6 步报告](benchmarks/step-6-service.md)。最终验收必须另行记录：

- Guild 安装：上传附歌词和不附歌词各一首 → 回执 → 出谱 → 6 个控件 → 3 种导出。
- User 安装：在私信执行同一流程；重启服务后再次使用原谱子的控件。
- 第二个账号点击控件应被拒绝；验证超配额、损坏文件、超长音频和纯音乐提示。
- 交互凭据过期、无发言权限、无法私信时，错误明确且 `/status job:` 能取回结果。
- 删除正在处理和已完成任务，确认服务端文件、数据库、旧控件的行为。
- 空队列时真实上传到消息出现不超过 5 分钟；干净机器部署耗时分别记录构建、下载和启动。

## 模型许可

项目代码 MIT，不等于所有权重都可商用。Demucs htdemucs 权重许可未明确，按非商业使用对待；BTC 与 lv-chordia 的权重没有单独声明许可。其它模型及训练数据说明见 [后端计划的许可证表](plans/backend.md#21-已接入的模型权重与许可证)。本步骤未新增模型，也未部署公共实例。
