#!/bin/bash
# Customer Service Response Examples
# Demonstrates different scenarios and template usage

echo "=== Customer Service Response Examples ==="
echo "Testing different customer service scenarios across multiple LLMs"
echo

# Example 1: Basic Support Request
echo "1. Testing Basic Support Request..."
python scripts/run_benchmarks.py \
  --prompt-file templates/customer_service/customer_service_response.txt \
  --prompt-variables '{
    "company_name": "TechSolutions Inc",
    "customer_type": "Premium", 
    "severity": "Medium",
    "customer_message": "My software keeps crashing when I export large datasets. This is affecting my daily work and I need a solution quickly.",
    "tone": "professional and helpful",
    "max_words": "150",
    "company_guidelines": "Premium customers receive priority support with 4-hour response time"
  }' \
  --models gpt-4o-mini,claude-3-haiku \
  --limit 1 \
  --output-format json,markdown \
  --output-dir ./results/examples/customer-service/basic

echo "✅ Basic support request completed"
echo

# Example 2: Escalation Scenario  
echo "2. Testing Escalation Scenario..."
python scripts/run_benchmarks.py \
  --prompt-file templates/customer_service/escalation_handling.txt \
  --prompt-variables '{
    "customer_name": "Alex Rodriguez",
    "customer_tier": "Enterprise",
    "issue_summary": "Critical data corruption affecting multiple team members",
    "escalation_reason": "Previous agents unable to resolve after 3 attempts",
    "urgency_level": "Critical",
    "previous_agents": "Sarah M., David K., Lisa T.",
    "attempted_solutions": "System restart, cache clearing, backup restoration - none successful"
  }' \
  --models gpt-4,claude-3-sonnet \
  --parallel \
  --metrics sentiment,coherence \
  --output-dir ./results/examples/customer-service/escalation

echo "✅ Escalation scenario completed"
echo

# Example 3: Technical Support
echo "3. Testing Technical Support..."
python scripts/run_benchmarks.py \
  --prompt-file templates/customer_service/customer_service_response.txt \
  --prompt-variables '{
    "company_name": "CloudHost Pro",
    "customer_type": "Developer",
    "severity": "Low", 
    "customer_message": "How do I configure SSL certificates for my staging environment? The documentation seems outdated and I am getting certificate errors.",
    "tone": "technical and helpful",
    "resolution_steps": "1) Use our updated SSL setup guide 2) Run cert-manager tool 3) Contact DevOps team if issues persist",
    "max_words": "200"
  }' \
  --models gpt-4o-mini,gemini-flash \
  --output-format csv \
  --output-dir ./results/examples/customer-service/technical

echo "✅ Technical support completed"
echo

# Example 4: Batch Testing Different Tones
echo "4. Testing Different Response Tones..."
for tone in "professional" "empathetic" "friendly"; do
  echo "   Testing tone: $tone"
  python scripts/run_benchmarks.py \
    --prompt-file templates/customer_service/customer_service_response.txt \
    --prompt-variables "{
      \"company_name\": \"ServiceCorp\", 
      \"customer_type\": \"Standard\", 
      \"severity\": \"Medium\",
      \"customer_message\": \"I've been waiting 3 days for a response to my ticket. This is very frustrating.\",
      \"tone\": \"$tone\", 
      \"max_words\": \"150\"
    }" \
    --models gpt-4o-mini \
    --limit 1 \
    --output-dir "./results/examples/customer-service/tones/$tone"
done

echo "✅ Tone testing completed"
echo

echo "=== All Customer Service Examples Completed ==="
echo "Results saved to: ./results/examples/customer-service/"
echo "View results:"
echo "  - Basic: cat results/examples/customer-service/basic/*.md"
echo "  - Escalation: cat results/examples/customer-service/escalation/*.json"
echo "  - Technical: cat results/examples/customer-service/technical/*.csv"
echo "  - Tones: ls results/examples/customer-service/tones/"