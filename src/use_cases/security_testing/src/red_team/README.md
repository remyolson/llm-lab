# Red Team Simulation Framework

This module provides advanced adversarial testing capabilities for Large Language Models through sophisticated red team simulation. It builds upon the existing security testing infrastructure to provide multi-step attack orchestration, domain-specific scenarios, and comprehensive attack analytics.

## Key Components

- **RedTeamSimulator**: Core orchestration engine for multi-step attack campaigns
- **AttackScenario**: Domain-specific attack templates and scenarios
- **AttackChain**: Sequential and parallel attack step execution
- **EvasionEngine**: Advanced evasion techniques and context manipulation
- **WorkflowEngine**: Automated and manual red team workflow management
- **CustomAttackFramework**: Framework for developing organization-specific attacks

## Usage

```python
from src.use_cases.security_testing.src.red_team import RedTeamSimulator
from src.use_cases.security_testing.src.attack_library import AttackLibrary

# Initialize red team simulator
attack_library = AttackLibrary.load_default()
simulator = RedTeamSimulator(attack_library)

# Run a domain-specific red team campaign
campaign = await simulator.run_campaign(
    model_interface=my_model,
    scenario="healthcare_phi_extraction",
    difficulty_level="advanced"
)

print(f"Campaign completed: {campaign.success_rate}% success rate")
```

## Attack Scenarios

The framework includes pre-built scenarios for:
- Healthcare (PHI extraction, medical advice manipulation)
- Financial Services (unauthorized trading, data extraction)
- Customer Service (social engineering, privilege escalation)
- General Purpose (jailbreaking, context manipulation)

## Advanced Features

- Multi-step attack chain orchestration
- Contextual state management across attack steps
- Real-time attack success scoring
- Adaptive evasion techniques
- Session isolation and result correlation
- Comprehensive analytics and reporting
