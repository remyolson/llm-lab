# Cost-Per-Quality Metrics Analysis

## Balanced Scenario

```
Model | Cost/Req | Cost/Acc | Cost/Sat | Efficiency Score
-----------------------------------------------------------------
google/gemini-1.5-flash   | $0.3750 | $0.0043 | $0.0893 | 2.43
anthropic/claude-3-5-haik | $0.7500 | $0.0083 | $0.1705 | 1.24
google/gemini-1.5-pro     | $3.1250 | $0.0340 | $0.6944 | 0.30
openai/gpt-4o             | $10.0000 | $0.1047 | $2.0833 | 0.09
```

**Best Choice**: google/gemini-1.5-flash (Efficiency Score: 2.43)

## Quality-Focused Scenario

```
Model | Cost/Req | Cost/Acc | Cost/Sat | Efficiency Score
-----------------------------------------------------------------
google/gemini-1.5-flash   | $0.3750 | $0.0043 | $0.0893 | 2.36
anthropic/claude-3-5-haik | $0.7500 | $0.0083 | $0.1705 | 1.21
google/gemini-1.5-pro     | $3.1250 | $0.0340 | $0.6944 | 0.30
openai/gpt-4o             | $10.0000 | $0.1047 | $2.0833 | 0.09
```

**Best Choice**: google/gemini-1.5-flash (Efficiency Score: 2.36)

## Speed-Focused Scenario

```
Model | Cost/Req | Cost/Acc | Cost/Sat | Efficiency Score
-----------------------------------------------------------------
google/gemini-1.5-flash   | $0.3750 | $0.0043 | $0.0893 | 2.51
anthropic/claude-3-5-haik | $0.7500 | $0.0083 | $0.1705 | 1.27
google/gemini-1.5-pro     | $3.1250 | $0.0340 | $0.6944 | 0.31
openai/gpt-4o             | $10.0000 | $0.1047 | $2.0833 | 0.09
```

**Best Choice**: google/gemini-1.5-flash (Efficiency Score: 2.51)

## Cost-Focused Scenario

```
Model | Cost/Req | Cost/Acc | Cost/Sat | Efficiency Score
-----------------------------------------------------------------
google/gemini-1.5-flash   | $0.3750 | $0.0043 | $0.0893 | 2.41
anthropic/claude-3-5-haik | $0.7500 | $0.0083 | $0.1705 | 1.23
google/gemini-1.5-pro     | $3.1250 | $0.0340 | $0.6944 | 0.30
openai/gpt-4o             | $10.0000 | $0.1047 | $2.0833 | 0.09
```

**Best Choice**: google/gemini-1.5-flash (Efficiency Score: 2.41)

