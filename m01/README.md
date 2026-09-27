# Module 1: Agent architecture and design

The first version of the course project: a support-desk agent for a fictional
electronics shop, plus a second "returns" agent it hands questions to over A2A.
Runs in free mode (Ollama or the NVIDIA API catalog). No GPU, no AWS.

## Install

From the repo root, with the Module 0 venv active:

```bash
pip install -r m01/requirements.txt
pip install -e m01/support_tools      # registers lookup_order and returns_policy with the toolkit
```

## Pick the model

Everything defaults to local Ollama (`llama3.2:3b`, from Module 0). For NVIDIA's API catalog:

```bash
export LLM_PROVIDER=nvidia                                   # for react_by_hand.py
export LLM_BASE_URL=https://integrate.api.nvidia.com/v1      # for the toolkit configs
export LLM_MODEL=nvidia/nemotron-3-super-120b-a12b           # or the model check_env.py told you to use
export LLM_API_KEY=$NVIDIA_API_KEY
```

## 1. ReAct by hand

```bash
python m01/react_by_hand.py "Where is my order A1001, and when will it arrive?"
```

Watch the loop: the model writes a Thought and an Action, the script runs the tool and
adds the Observation, and the model continues until it writes a Final Answer.

## 2. The same agent in NeMo Agent Toolkit

```bash
nat run --config_file m01/configs/support_agent.yml --input "What is the status of order A1003?"
```

## 3. Two agents over A2A

```bash
# terminal 1: publish the returns agent
nat a2a serve --config_file m01/configs/returns_agent.yml --port 11000 --name returns_agent \
  --description "Answers questions about returns and refunds"
# terminal 2: look at its Agent Card, then let the support agent call it
curl -s http://localhost:11000/.well-known/agent-card.json
nat run --config_file m01/configs/support_with_returns.yml \
  --input "Order A1002 was delivered 3 days ago. Can I still return it, and how soon is the refund?"
```

## Check

With the returns agent still running in terminal 1:

```bash
python m01/check.py
```

Small local models don't follow the ReAct format every time. If a check fails, run it
again; if it keeps failing, try NVIDIA mode or a larger local model (`OLLAMA_MODEL=...`
for step 1, `LLM_MODEL=...` for the toolkit).

The orders and returns policy in `m01/data/` are made up for the course.
