1. 安装 Docker Desktop：点击下载软件安装 https://www.docker.com/get-started/
2. 打开 PowerShell。
3. 下载安装脚本，命令如下：Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat
4. 运行下载的脚本：standalone.bat start （也可打开 Docker Desktop 运行安装的 Milvus 相关组件）
5. 运行 python src\app.py
6. 访问 http://127.0.0.1:7860/ 开始问答