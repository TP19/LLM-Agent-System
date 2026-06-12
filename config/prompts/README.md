# Agent Prompts

**Status**: ✅ Implemented
**Version**: 1.0

---

## Overview

This directory contains externalized prompt templates for all LLM-Agent-System agents. Externalizing prompts allows for:

- **Easy Tuning**: Modify prompts without changing code
- **Version Control**: Track prompt changes separately
- **A/B Testing**: Test different prompt strategies
- **Consistency**: Ensure all agents use standardized formats
- **Maintainability**: Centralized prompt management

---

## File Structure

```
config/prompts/
├── README.md           # This file
├── triage.yaml         # Triage agent prompts
├── security.yaml       # Security agent prompts
├── executor.yaml       # Executor/Operator agent prompts
├── coder.yaml          # Coder agent prompts
└── oracle.yaml         # Oracle agent prompts
```

---

## Prompt File Format

Each YAML file contains prompts for a specific agent:

```yaml
# Agent Name Prompts
# Description of agent's purpose

system_prompt: |
  Multi-line system prompt
  that defines the agent's role
  and responsibilities

specific_task_prompt: |
  Template prompt with {placeholders}
  for variable substitution

  Example: {variable_name}
```

---

## Usage in Agents

### Loading Prompts

```python
import yaml
from pathlib import Path

class MyAgent(BaseAgent):
    def __init__(self, model_manager):
        super().__init__("my_agent", model_manager)

        # Load prompts
        prompt_file = Path("config/prompts/my_agent.yaml")
        with open(prompt_file) as f:
            self.prompts = yaml.safe_load(f)

    def process_request(self, request):
        # Build prompt using template
        prompt = self.build_prompt(
            self.prompts['task_prompt'],
            request=request,
            context="additional context"
        )

        # Generate response
        response = self.retry_generate(prompt)

        # Parse JSON response
        result = self.parse_json_response(response)
        return result
```

### Using BaseAgent Helper

```python
# BaseAgent provides build_prompt() helper
prompt = self.build_prompt(
    template=self.prompts['classification_prompt'],
    request=user_input,
    categories="coding, sysadmin, debugging"
)
```

---

## Prompt Templates

### Triage Agent (`triage.yaml`)

**Purpose**: Classify and route user requests

**Prompts**:
- `system_prompt` - Defines classification categories
- `classification_prompt` - Main classification task
- `reclassification_prompt` - Re-classify based on feedback
- `multi_agent_prompt` - Identify multi-agent workflows

**Categories**:
- sysadmin, coding, summarization, knowledge, security, docker, general

### Security Agent (`security.yaml`)

**Purpose**: Security analysis and risk assessment

**Prompts**:
- `system_prompt` - Security analysis role
- `analysis_prompt` - Assess request security
- `command_review_prompt` - Review command safety
- `vulnerability_assessment_prompt` - Identify vulnerabilities
- `remediation_prompt` - Provide security fixes

**Risk Levels**:
- low, medium, high

### Executor Agent (`executor.yaml`)

**Purpose**: System operations and command execution

**Prompts**:
- `system_prompt` - Executor role
- `execution_plan_prompt` - Create execution plan
- `command_generation_prompt` - Generate commands
- `error_analysis_prompt` - Analyze errors
- `result_summary_prompt` - Summarize results

### Coder Agent (`coder.yaml`)

**Purpose**: Software development and code generation

**Prompts**:
- `system_prompt` - Coder role and best practices
- `code_generation_prompt` - Generate new code
- `code_review_prompt` - Review code quality
- `debugging_prompt` - Debug issues
- `refactoring_prompt` - Refactor code
- `architecture_design_prompt` - Design systems

### Oracle Agent (`oracle.yaml`)

**Purpose**: High-level planning and coordination

**Prompts**:
- `system_prompt` - Strategic planning role
- `task_decomposition_prompt` - Break down complex tasks
- `strategy_planning_prompt` - Develop strategies
- `agent_coordination_prompt` - Coordinate multiple agents
- `problem_solving_prompt` - Analyze and solve problems

---

## Best Practices

### 1. Clear Structure

```yaml
prompt_name: |
  Clear description of what the agent should do

  Input Format:
  - Describe expected inputs

  Output Format:
  - Describe expected output format (JSON, text, etc.)

  Example:
  - Provide example if helpful
```

### 2. Variable Placeholders

Use `{variable_name}` for template variables:

```yaml
analysis_prompt: |
  Analyze this: {item}
  Context: {context}
  Constraints: {constraints}
```

### 3. JSON Output Format

Specify exact JSON structure expected:

```yaml
task_prompt: |
  Provide JSON response:
  {{
    "key": "value",
    "array": ["item1", "item2"]
  }}
```

Note: Use double braces `{{` and `}}` in YAML to escape for Python's `.format()`.

### 4. Clear Instructions

Be explicit about:
- What to do
- What format to use
- What to include/exclude
- Edge cases to consider

### 5. System Prompts

Include in system prompts:
- Agent's role and responsibilities
- Output format expectations
- Quality standards
- Example responses

---

## Modifying Prompts

### Testing Prompt Changes

1. **Backup Original**:
   ```bash
   cp config/prompts/agent.yaml config/prompts/agent.yaml.bak
   ```

2. **Edit Prompt**:
   ```bash
   vim config/prompts/agent.yaml
   ```

3. **Test**:
   ```bash
   python -c "
   from agents.agent_name import AgentClass
   agent = AgentClass(model_manager)
   result = agent.process('test request')
   print(result)
   "
   ```

4. **Compare Results**:
   - Test with multiple inputs
   - Check JSON parsing still works
   - Verify output quality

5. **Rollback if Needed**:
   ```bash
   mv config/prompts/agent.yaml.bak config/prompts/agent.yaml
   ```

### Prompt Engineering Tips

1. **Be Specific**: Vague prompts produce vague results
2. **Use Examples**: Show the model what you want
3. **Set Constraints**: Define boundaries and limitations
4. **Request Structured Output**: JSON is easier to parse than free text
5. **Iterate**: Test, measure, refine

---

## Versioning

Track prompt versions in git:

```bash
# View prompt changes
git log config/prompts/triage.yaml

# Compare versions
git diff HEAD~1 config/prompts/triage.yaml

# Revert to previous version
git checkout HEAD~1 -- config/prompts/triage.yaml
```

---

## Migration Guide

### For Existing Agents

**Before** (hardcoded prompts):
```python
class TriageAgent(BaseAgent):
    def classify(self, request):
        prompt = f"You are a triage agent. Classify: {request}"
        response = self.generate(prompt)
        return self.parse_json(response)
```

**After** (externalized prompts):
```python
class TriageAgent(BaseAgent):
    def __init__(self, model_manager):
        super().__init__("triage", model_manager)

        # Load prompts
        with open("config/prompts/triage.yaml") as f:
            self.prompts = yaml.safe_load(f)

    def classify(self, request):
        # Use template
        prompt = self.build_prompt(
            self.prompts['classification_prompt'],
            request=request
        )
        response = self.retry_generate(prompt)
        return self.parse_json_response(response)
```

---

## Future Enhancements

Planned improvements:

1. **Prompt Variants**: Multiple versions for A/B testing
2. **Language-Specific**: Prompts optimized for different models
3. **Dynamic Loading**: Hot-reload prompts without restart
4. **Metrics**: Track which prompts perform best
5. **Prompt Library**: Shared prompts across agents

---

## Troubleshooting

### Prompt Not Found

**Error**: `FileNotFoundError: config/prompts/agent.yaml`

**Solution**:
```bash
# Check file exists
ls config/prompts/

# Check working directory
pwd

# Use absolute path in code
from pathlib import Path
prompt_file = Path(__file__).parent.parent / "config" / "prompts" / "agent.yaml"
```

### JSON Parsing Fails

**Error**: `JSONDecodeError: Expecting value`

**Solution**:
1. Check prompt requests JSON format explicitly
2. Use `parse_json_response()` with default fallback
3. Add examples to prompt showing exact format
4. Use double braces `{{` `}}` for literal braces in YAML

### Variables Not Substituted

**Error**: `KeyError: 'variable_name'`

**Solution**:
```python
# Use build_prompt() which handles missing variables
prompt = self.build_prompt(template, **vars)

# Or provide all required variables
prompt = template.format(
    request=request,
    context=context or "none"  # Provide defaults
)
```

---

## See Also

- `core/base_agent.py` - BaseAgent with helper methods
- `PHASE_11_5_COMPLETION.md` - Implementation summary
- `docs/PHASE_11_5_SPECIFICATION.md` - Full specification
- Agent implementation files in `agents/`

---

**Status**: externalized prompts for 5 core agents (triage, security, executor, coder, oracle). Additional agents can follow the same pattern.
