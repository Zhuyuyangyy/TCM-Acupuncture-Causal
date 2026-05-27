# TCM-Acupuncture-Causal

针灸真实世界因果推断与个体化穴位组合推荐系统

## 核心创新点

1. **针灸因果推断** — 将 target trial emulation 应用于针灸疗效评价
2. **个体化穴位推荐** — 基于 uplift modeling 的穴位组合优化
3. **关联规则挖掘** — Apriori/FP-Growth 发现穴位配伍规律
4. **反事实分析** — 估计不同穴位组合的个体化治疗效果

## 快速开始

```bash
pip install -e .
uvicorn backend.main:app --port 8020 --reload
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/protocol` | GET/POST | 针灸试验协议管理 |
| `/api/analysis/causal` | POST | 因果效应估计 |
| `/api/analysis/uplift` | POST | 个体化效果估计 |
| `/api/analysis/rules` | POST | 穴位关联规则挖掘 |

## License

MIT
