# 通义千问模型集成项目

## 项目简介

这个项目展示了如何将阿里云通义千问(Qwen)大语言模型集成到Python应用程序中，使用DashScope SDK和LangChain框架。

## 环境设置

### 1. 创建虚拟环境

**使用venv (Python官方)**:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**使用Anaconda**:

```bash
conda create -n qwen-env python=3.9
conda activate qwen-env
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 配置环境变量

### 获取DASHSCOPE_API_KEY

1. 访问 [阿里云通义千问控制台](https://dashscope.console.aliyun.com/)
2. 注册或登录您的阿里云账号
3. 创建并获取您的API密钥
4. 确保您有足够的API调用额度

### 设置API密钥

1. 在 `.env` 文件中设置您的 API 密钥：
   ```
   # 通义千问API密钥
   DASHSCOPE_API_KEY=您的DASHSCOPE_API_KEY
   ```

2. 或者在系统环境变量中设置：
   - **Windows**: 
     ```powershell
     setx DASHSCOPE_API_KEY "您的DASHSCOPE_API_KEY"
     ```
     设置后需要重启命令提示符或PowerShell
   - **macOS/Linux**: 
     ```bash
     export DASHSCOPE_API_KEY="您的DASHSCOPE_API_KEY"
     ```
     或者将其添加到 `~/.bashrc` 或 `~/.zshrc` 文件中使其永久生效

### 验证API密钥设置

在运行程序前，您可以验证API密钥是否正确设置：

```powershell
# Windows
set DASHSCOPE_API_KEY

# macOS/Linux
echo $DASHSCOPE_API_KEY
```

> **注意**: 请不要将您的API密钥提交到代码仓库中。确保 `.env` 文件已添加到 `.gitignore` 中。

## 通义千问模型配置

### 支持的模型

- `qwen-plus`: 标准模型，适合大多数场景
- `qwen-max`: 高级模型，提供更好的性能
- `qwen-turbo`: 快速响应模型，适合实时交互
- `qwen1.5-72b-chat`: 1.5版本72B参数模型
- `qwen1.5-14b-chat`: 1.5版本14B参数模型

### 修改模型类型

要更改使用的模型，请修改 `main.py` 文件中的 `llm_invoke` 函数中的模型参数：

```python
def llm_invoke(prompt):
    # 处理不同类型的输入
    # ...
    return generate_answer(prompt_text, "qwen-plus")  # 可以改为其他模型
```

## 运行项目

```bash
python main.py
```

## 常见问题排查

### 1. API密钥错误

如果出现 `InvalidApiKey` 错误，请检查：
- API密钥是否正确复制
- API密钥是否过期
- 账户是否有足够的调用额度

### 2. 模型调用失败

如果模型调用失败，请检查：
- 网络连接是否正常
- DashScope服务是否正常运行
- 输入内容是否符合模型要求

### 3. 依赖问题

如果遇到依赖相关错误，请尝试：
```bash
pip install --upgrade -r requirements.txt
```

## 环境变量安全注意事项

1. **不要硬编码API密钥**：永远不要在代码中直接写入API密钥
2. **使用环境变量**：始终通过环境变量或.env文件管理敏感信息
3. **不要提交.env文件**：确保将.env文件添加到.gitignore中
4. **定期轮换密钥**：定期更新您的API密钥以提高安全性

## 许可证

MIT License