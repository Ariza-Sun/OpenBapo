# 第 6 步：Discord Bot

状态：本地实现与验证完成，等待真实 Discord / 部署验收（2026-11-16）。负责人已授权主动推进，需求明确的工程决定直接实施。

## 实现决定

- discord.py 2.x + Gateway，注册全局命令，同时允许服务器安装、用户安装和私信。
- 单实例、单 GPU 串行处理；SQLite WAL 保存任务、配额、分析结果和调整参数。
  当前无需增加 API、Redis、独立数据库或对象存储。进程锁防止误开两个实例。
- 每首歌在独立子进程中运行现有分析流程，父进程负责截止时间、取消和临时目录清理。
  原音频、临时分轨、用户附上的歌词文本在处理结束后删除；结构化结果保留 30 天。
- 每人每天 5 首，按 UTC 日重置；格式、大小、时长校验失败不计数，已入队任务计数。
  删除项目不重置当日配额；计数仅保留当日不可直接识别用户的 HMAC 标识。
- 队列也计入提交后的 15 分钟硬超时。预计等待时间属于估计，不能保证繁忙时 5 分钟出谱。
- 控件按前端计划第 5.2 节的 **6 个** 实现：演唱调、指法调、节奏型、简化、导出、重置。
  路线图的“7 个”是旧计数。节奏型按当前拍号筛选，保留自动和无；无需二级菜单。
- 调整只使用已保存的 Analysis，重启后恢复控件，所有操作在服务端再次核对发起者。
- 交互先 defer；有效期内使用交互 webhook，过期后尝试原频道，再尝试私信。
  投递失败仍保留结果，用户可以通过 `/status` 重取。
- 文案中英双语；初版保留现有谱面中文标签。未知调性、无歌词和低置信和弦明确提示。

## 验证与剩余验收

本地验证覆盖持久化、并发配额、越权、重启、取消、超时、清理以及真实编配和导出。
必须另行完成 Discord 测试服务器和用户级安装私信验收，才将第 6 步标为完成。
干净机器部署耗时和真实上传到出谱耗时也须实际记录，不能用核心流水线耗时替代。

本轮已完成：

- 4 个斜杠命令与消息右键入口、中英文命令注册、服务器 / 用户安装上下文；6 个持久控件、只允许发起者调整或导出。
- SQLite 原子配额与 FIFO 队列；单实例锁；子进程硬超时、删除取消、重启恢复、排队期间超时清理。
- 附件检查、8 分钟限制、禁止解码器打开网络协议；损坏容器低报时长时也限制解码长度。
- 原音频与内部临时分轨删除；30 天结果保留、按日配额清理、短期交互凭据清理。
- 交互 webhook、频道和私信的投递路径，以及投递失败后 `/status job:` 取回。
- `Dockerfile`、`compose.yaml`、模型预热脚本和 [自部署文档](../self-hosting.md)。

验证记录：

- `pytest --junitxml=/tmp/openbapo-step6-tests.xml`：全量 660 项通过，7 项模型标记测试未在该命令运行。
- 最后增加排队文件过期 / 真实超长音频测试并调整解码上限后，运行
  `pytest tests/test_audio.py tests/test_service.py tests/test_bot.py`：43 项通过，包含全部 25 项服务 / Discord 测试。
- `ruff check .`、`ruff format --check .`、`mypy`（69 个源文件）、`git diff --check` 均通过。
- `docker compose --env-file .env.example config --no-env-resolution --quiet` 和入口 shell 语法检查通过。
- `scripts/warm_service_models.py --device cuda` 用现有缓存成功加载全部模型；未声称从空缓存下载已验证。
- 真实 4 分钟音频通过新 worker：最终代码本地提交到预览 **49.22 秒**，演唱调调整到新预览 **1.32 秒**。
  该轮包含独立子进程和模型初始化，不含权重下载 / Discord 网络；来源指纹和分阶段计时见
  [脚本报告](../benchmarks/step-6-service.md)。此前实现迭代的计时 50.27 秒不作为最终代码结果。

仍未完成：

- 工作区没有 Discord Token 配置；未登录真实应用，也未发送测试消息。需负责人指定应用配置位置、测试频道，并授权测试消息后完成服务器 / 私信验收。
- 本机 Docker Engine socket 拒绝当前用户访问，尚未构建镜像或运行容器。配置可解析不等于镜像可构建。
- 未在干净机器验证 30 分钟部署，未测 Discord 上传到出谱的真实总时延。
- 改动保留在工作区，未提交 / 推送，远端 CI 尚未跑本轮内容。

因此第 6 步维持“进行中”，M1 不提前标为完成。

## 接口依据

- [Discord 交互响应](https://docs.discord.com/developers/interactions/receiving-and-responding)：及时回应、交互 token 有效期、附件限额。
- [用户级安装](https://docs.discord.com/developers/tutorials/developing-a-user-installable-app)：安装位置与交互上下文分别配置。
- [discord.py 交互接口](https://discordpy.readthedocs.io/en/stable/interactions/api.html)：持久 View、命令本地化和安装上下文。
