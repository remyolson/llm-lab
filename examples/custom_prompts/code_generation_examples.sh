#!/bin/bash
# Code Generation Examples
# Demonstrates different complexity levels and programming tasks

echo "=== Code Generation Examples ==="
echo "Testing code generation across different complexity levels and tasks"
echo

# Example 1: Beginner Level Function
echo "1. Testing Beginner Level Function..."
python scripts/run_benchmarks.py \
  --prompt-file templates/code_generation/code_generation.txt \
  --prompt-variables '{
    "function_name": "calculate_circle_area",
    "function_purpose": "Calculate the area of a circle given its radius",
    "complexity_level": "beginner",
    "input_parameters": "radius (float)",
    "return_type": "float",
    "include_examples": true,
    "style_preferences": "Clear variable names with detailed comments explaining each step"
  }' \
  --models gpt-4o-mini,claude-3-haiku \
  --limit 1 \
  --output-format markdown \
  --output-dir ./results/examples/code-generation/beginner

echo "✅ Beginner function completed"
echo

# Example 2: Intermediate Level Data Processing
echo "2. Testing Intermediate Level Data Processing..."
python scripts/run_benchmarks.py \
  --prompt-file templates/code_generation/code_generation.txt \
  --prompt-variables '{
    "function_name": "process_csv_data",
    "function_purpose": "Process CSV file and extract specific columns with validation",
    "complexity_level": "intermediate",
    "input_parameters": "csv_file_path (str), required_columns (List[str]), output_format (str)",
    "return_type": "Dict[str, List[Any]]",
    "additional_requirements": "Handle missing files, validate column existence, support both dict and list output formats, include error handling for malformed CSV",
    "include_examples": true,
    "include_tests": true,
    "style_preferences": "Type hints, docstrings, PEP 8 compliance"
  }' \
  --models gpt-4,claude-3-sonnet \
  --parallel \
  --metrics all \
  --output-dir ./results/examples/code-generation/intermediate

echo "✅ Intermediate function completed"
echo

# Example 3: Advanced Algorithm Implementation
echo "3. Testing Advanced Algorithm Implementation..."
python scripts/run_benchmarks.py \
  --prompt-file templates/code_generation/code_generation.txt \
  --prompt-variables '{
    "function_name": "find_shortest_path",
    "function_purpose": "Find shortest path between nodes using A* algorithm",
    "complexity_level": "advanced",
    "input_parameters": "graph (Dict[str, Dict[str, float]]), start (str), goal (str), heuristic_func (Callable)",
    "return_type": "Tuple[List[str], float]",
    "additional_requirements": "Implement A* pathfinding with configurable heuristic, handle disconnected graphs, optimize for performance, include path reconstruction",
    "include_examples": true,
    "include_tests": true,
    "style_preferences": "Production-ready code with comprehensive documentation, type hints, and performance considerations"
  }' \
  --models gpt-4,claude-3-opus \
  --timeout 120 \
  --output-dir ./results/examples/code-generation/advanced

echo "✅ Advanced algorithm completed"
echo

# Example 4: Code Review Example
echo "4. Testing Code Review..."
python scripts/run_benchmarks.py \
  --prompt-file templates/code_generation/code_review.txt \
  --prompt-variables '{
    "code": "def login_user(username, password):\n    if username == \"admin\" and password == \"123456\":\n        return True\n    else:\n        return False\n\ndef get_user_data(user_id):\n    query = \"SELECT * FROM users WHERE id = \" + str(user_id)\n    return database.execute(query)",
    "language": "python",
    "focus_areas": "security, best practices, maintainability",
    "security_focus": "SQL injection, authentication vulnerabilities, password handling"
  }' \
  --models gpt-4,claude-3-sonnet \
  --parallel \
  --output-format json,markdown \
  --output-dir ./results/examples/code-generation/review

echo "✅ Code review completed"
echo

# Example 5: Batch Testing Different Function Types
echo "5. Testing Different Function Types..."
declare -a functions=(
  "sort_algorithm:Implement merge sort algorithm:intermediate"
  "validate_email:Email address validation with regex:beginner"
  "cache_decorator:Python decorator for function caching:advanced"
  "binary_search:Binary search implementation:intermediate"
)

for func in "${functions[@]}"; do
  IFS=':' read -r name purpose complexity <<< "$func"
  echo "   Testing: $name ($complexity)"

  python scripts/run_benchmarks.py \
    --prompt-file templates/code_generation/code_generation.txt \
    --prompt-variables "{
      \"function_name\": \"$name\",
      \"function_purpose\": \"$purpose\",
      \"complexity_level\": \"$complexity\",
      \"include_examples\": true,
      \"style_preferences\": \"Clean, readable code with appropriate comments\"
    }" \
    --models gpt-4o-mini \
    --limit 1 \
    --output-dir "./results/examples/code-generation/batch/$name"
done

echo "✅ Batch function generation completed"
echo

# Example 6: A/B Testing Code Styles
echo "6. Testing Different Code Styles..."
python scripts/run_benchmarks.py \
  --prompt-file templates/code_generation/code_generation.txt \
  --prompt-variables '{
    "function_name": "fibonacci",
    "function_purpose": "Calculate fibonacci number recursively",
    "complexity_level": "intermediate",
    "style_preferences": "Functional programming style with minimal comments"
  }' \
  --models gpt-4 \
  --limit 2 \
  --output-dir ./results/examples/code-generation/style-test/functional

python scripts/run_benchmarks.py \
  --prompt-file templates/code_generation/code_generation.txt \
  --prompt-variables '{
    "function_name": "fibonacci",
    "function_purpose": "Calculate fibonacci number recursively",
    "complexity_level": "intermediate",
    "style_preferences": "Object-oriented style with detailed documentation"
  }' \
  --models gpt-4 \
  --limit 2 \
  --output-dir ./results/examples/code-generation/style-test/oop

echo "✅ Style comparison completed"
echo

echo "=== All Code Generation Examples Completed ==="
echo "Results saved to: ./results/examples/code-generation/"
echo "View results:"
echo "  - Beginner: cat results/examples/code-generation/beginner/*.md"
echo "  - Intermediate: cat results/examples/code-generation/intermediate/*.json"
echo "  - Advanced: cat results/examples/code-generation/advanced/*.json"
echo "  - Review: cat results/examples/code-generation/review/*.md"
echo "  - Batch: ls results/examples/code-generation/batch/"
echo "  - Style comparison: diff results/examples/code-generation/style-test/functional/*.json results/examples/code-generation/style-test/oop/*.json"
