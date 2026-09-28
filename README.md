# NCP-AAI course labs

Hands-on labs for the free **NVIDIA-Certified Professional: Agentic AI (NCP-AAI)** exam-prep
course by Cloud Brewery Academy. Videos are free on YouTube; this repo holds the code.

Independent course. Not affiliated with or endorsed by NVIDIA.

## Two ways to run the labs

| Mode | Runs on | What you need |
|---|---|---|
| Free | Your laptop | Python 3.10+, and either Ollama (free, local, no key) or an NVIDIA API key from build.nvidia.com |
| AWS GPU | One `g6e.xlarge` in `us-east-1` | An AWS account, GPU quota, a budget alarm |

Start with **[setup/](setup/README.md)** (Module 0). Each later module lives on its own
branch (`m01-react` ... `m10-hitl`), so you can join at any module.

## Cost safety (AWS mode)

- The GPU instance terminates itself 55 minutes after it boots.
- `python setup/aws_lab.py down` deletes everything a lab created. Run it as soon as
  you finish.
- Prices change: check the AWS pricing page for `g6e.xlarge` before you start.
