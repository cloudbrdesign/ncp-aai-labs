# Module 0: environment setup

## 1. Free mode (no GPU)

```bash
git clone https://github.com/cloudbrdesign/ncp-aai-labs.git
cd ncp-aai-labs
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r setup/requirements.txt
```

Then pick **one** model provider. Every lab gets its model from `setup/llm.py`, so the
lab code is identical either way.

**Option A: NVIDIA API catalog** (hosted models, needs an API key)

```bash
export NVIDIA_API_KEY=nvapi-...      # build.nvidia.com > a model page > Get API Key
python setup/check_env.py --mode api
```

Keep your key out of code and out of git. `.env` is already in `.gitignore`.
Hosted models change: if the default is retired (410) or not open to your key (403), the
check tries the next one and tells you which to use (`export NVIDIA_MODEL=...`).

**Option B: Ollama** (a free open model on your own machine, no key, no approval)

```bash
# install Ollama from https://ollama.com and start it, then:
ollama pull llama3.2:3b
export LLM_PROVIDER=ollama
python setup/check_env.py --mode api
```

Both options also check that the model can call tools, which the agent labs need.

## 2. AWS GPU mode

Needs AWS credentials for `us-east-1` (`aws configure`, or an `AWS_PROFILE`).

```bash
python setup/aws_lab.py quota                          # G and VT quota must be 4+ vCPUs
python setup/aws_lab.py quota --request 8              # if it is 0; AWS reviews the request
python setup/aws_lab.py budget --email you@example.com # monthly budget with email alerts
python setup/aws_lab.py up                             # 1x g6e.xlarge, self-terminates after 55 min
python setup/check_env.py --mode aws
python setup/aws_lab.py down                           # delete everything. Do not skip.
```

What `up` creates (one CloudFormation stack, `ncp-aai-gpu-lab`): a security group with
no inbound access, a launch template, and one instance from the AWS Deep Learning Base
GPU AMI (Ubuntu 22.04). The instance prints its GPU, driver, CUDA, Docker and NVIDIA
Container Toolkit versions to its console, then shuts itself down (which terminates it)
55 minutes after every boot. `down` deletes the stack.

### Module 5: a NIM on the lab instance

`up --lab m05` adds three things to the same stack: an instance role (path `/ncp-aai/`, with
the `AmazonSSMManagedInstanceCore` managed policy and read access to one SSM parameter), a
150 GiB root disk for the NIM image and weights, and user data that pulls
`nvcr.io/nim/meta/llama-3.1-8b-instruct:2.0.13`, lists its model profiles, starts it on port
8000 with tool calling on, waits for `/v1/health/ready` and prints what happened to the
console. `aws_lab.py` shows those lines as they arrive. Because the stack creates an IAM role,
`create_stack` is called with `CAPABILITY_IAM`.

```bash
python setup/aws_lab.py ngc-key          # your NGC API key, typed hidden -> SSM SecureString /ncp-aai/ngc-api-key
python setup/aws_lab.py up --lab m05
python setup/aws_lab.py tunnel           # localhost:8000 -> the NIM, through Session Manager
python setup/aws_lab.py down
```

The tunnel needs the AWS CLI and the Session Manager plugin on your machine; no inbound port
is opened. `setup/iam/lab-policy.json` lists the permissions every command here needs
(us-east-1 only; IAM actions limited to the `/ncp-aai/` path; `ssm:StartSession` only on
instances tagged `Project=ncp-aai-labs` with the port-forwarding document). Attach it (or
equivalent) to the user or role you run the labs with.

## Tested with

| Item | Value |
|---|---|
| Date | 2026-09-26 (AWS mode), 2026-09-27 (free mode) |
| Machine | macOS, Python 3.12.2 |
| Packages | see `requirements.txt` |
| Free mode | Ollama, `llama3.2:3b`: answer and tool call pass |
| AMI | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260925 |
| Instance | g6e.xlarge (1x NVIDIA L40S, 46068 MiB), us-east-1 |
| Driver / CUDA SDK / Docker / Container Toolkit | 595.91.07 / 13.2 / 29.8.1 / 1.20.1 |
| OS on the instance | Ubuntu 22.04.5 LTS |
