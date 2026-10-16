# Licensing

This repository uses two licences: one for software and one for educational content. The licence texts in
[`LICENSES/`](LICENSES/) are authoritative; this page only explains which one applies where.

| Material | Licence | Licence text |
|---|---|---|
| Software and source code | **Apache License 2.0** (`Apache-2.0`) | [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt) |
| Educational content and documentation | **Creative Commons Attribution 4.0 International** (`CC-BY-4.0`) | [LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt) |

Copyright 2026 Cloud Brewery Academy.

## Software: Apache-2.0

Original source code, and configuration that is executed or read by code. For example:
- the Python lab code and checks in `setup/` and `m01/` to `m11/`, including tests;
- configuration read by code: model and app configs, prompt templates, guardrail and rail definitions
  (`.yml`, `.yaml`, `.toml`, `.json`, `.co`);
- Kubernetes manifests in `m07/k8s/` (except the third-party sample listed below);
- CI workflows (`.github/`).

## Educational content and documentation: CC-BY-4.0

Original written and visual material. For example:
- READMEs and lab guides, checklists and notes;
- diagrams (`.mmd`) and images;
- the fictional shop's teaching data: product manuals (`m04/data/manuals/`), questions, tickets and the
  attack, benign and pair sets (`.jsonl`).

Code keeps Apache-2.0 even when it contains comments or docstrings. A suggested attribution: *"Based on material
from Cloud Brewery Academy's NCP-AAI labs (CC BY 4.0)."*

## Names and marks

The licences cover this repository's copyrightable content. They do not permit using the Cloud Brewery Academy name,
logo or marks as your own source identifier, or in a way that implies Cloud Brewery Academy endorsement. Ordinary
attribution, as above, is welcome.

## Third-party material and scope

- **`m07/k8s/nimservice.yaml`** is the LLM sample manifest from the NVIDIA NIM Operator documentation ("Managing NIM
  Services"), with comments added. It remains under NVIDIA's terms; no ownership is claimed.
- **Dependencies, models and product names** (including NVIDIA, AWS, Ollama and the Python packages in
  `requirements.txt`) remain under their own terms. No ownership of them is claimed.
- **Scope:** these licences apply to this repository only; they do not extend to any other Cloud Brewery Academy
  repository, course, product or service.
- **No warranty:** both licences are provided without warranty (see the licence texts). The labs show results under
  tested conditions; they are not a production security guarantee.

Independent course. Not affiliated with or endorsed by NVIDIA.
