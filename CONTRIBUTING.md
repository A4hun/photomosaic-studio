# 贡献指南

欢迎提 issue 和 PR！

## 跑测试

```powershell
pip install -e ".[dev,gui]"
python -m pytest tests -q
```

## 代码风格

- 注释、docstring、错误消息、GUI 文案用**中文**（项目面向中文用户）
- 类型注解：公共 API 加注解，内部实现宽松
- 新功能带测试；修 bug 先写复现测试再修

## PR 约定

- 所有测试必须通过（CI 会跑 Windows + Ubuntu × Python 3.11–3.13）
- 改动行为要在 CHANGELOG.md 记一笔
- 大改动先在 issue 里讨论

## 报告 Bug

提 issue 时请附：

- `photomosaic --version` 输出
- 复现步骤
- 错误日志：`%LOCALAPPDATA%\photomosaic-studio\error.log`（GUI 崩溃时自动生成）
- 脱敏后的最小复现素材（别直接贴私人照片）

## 逆向与合规

本项目是 clean-room 独立实现：算法结构参考了对经典商业拼图软件的技术分析，但代码从零编写，未复制任何原始厂商代码或资源。提交代码时请确保不引入任何第三方版权素材。
