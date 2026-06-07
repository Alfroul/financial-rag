# ADR 0004: 从 SiliconFlow Qwen3-8B 迁移到小米 MiMo-V2-Pro

## 状态

已批准

## 上下文

现有项目使用 SiliconFlow 提供的 Qwen3-8B 免费模型，2026 年在简历项目中的说服力不足：
- 8B 参数规模太小，Faithfulness 基线只有 0.80
- SiliconFlow 是中间商平台，不是模型提供方，简历上缺乏辨识度
- 面试官会质疑"为什么不用更强的模型"

## 决策

迁移到小米 MiMo-V2-Pro，理由：
1. 模型能力更强，Faithfulness 预期可达 0.90+
2. 小米是国内头部 AI 厂商，简历上更有辨识度
3. API 兼容 OpenAI 格式，迁移成本低
4. 有免费额度，适合开发阶段

## 后果

- 正面：更好的评测数据、更好的简历辨识度、面试可讲小米生态
- 负面：依赖小米 API 可用性，需要 MIMO_API_KEY
- 缓解：保留 `base_url` 配置，可随时切换回 SiliconFlow 或其他 OpenAI 兼容 API
