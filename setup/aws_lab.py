"""AWS helper for the NCP-AAI course labs (AWS-GPU mode).

    python setup/aws_lab.py quota                 # show your GPU instance quota
    python setup/aws_lab.py quota --request 8     # ask AWS to raise it to 8 vCPUs
    python setup/aws_lab.py budget --email you@example.com [--limit 10]
    python setup/aws_lab.py up                    # launch one GPU instance, run checks
    python setup/aws_lab.py ngc-key               # (Module 5) store your NGC key in SSM, typed hidden
    python setup/aws_lab.py up --lab m05          # (Module 5) the same, plus a Llama 3.1 8B NIM
    python setup/aws_lab.py tunnel                # (Module 5) NIM on http://localhost:8000 via Session Manager
    python setup/aws_lab.py status
    python setup/aws_lab.py down                  # delete everything `up` created

Cost rules built in:
  * the instance terminates itself MAX_MINUTES after boot, even if you forget `down`;
  * `down` deletes the whole stack and waits until AWS confirms it is gone.
Check current GPU prices on the AWS pricing page before you run `up`.
"""
import argparse
import datetime as dt
import pathlib
import sys
import time

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"                      # course standard (all labs)
# `up` falls back to these regions, in order, when a region has no GPU capacity or quota.
# Each region has its own GPU quota: check it with `quota --region <name>`.
REGIONS = ["us-east-1", "us-east-2", "us-west-2"]
STACK = "ncp-aai-gpu-lab"
INSTANCE_TYPE = "g6e.xlarge"              # 1x NVIDIA L40S, 44 GiB GPU memory, 4 vCPUs
INSTANCE_VCPUS = 4
MAX_MINUTES = 55                          # hard cap, enforced on the instance itself
BUDGET_NAME = "ncp-aai-labs"
# EC2 quota "Running On-Demand G and VT instances" (counted in vCPUs)
QUOTA_CODE = "L-DB2E81BA"
# Public SSM parameter holding the latest Deep Learning Base GPU AMI (Ubuntu 22.04)
DLAMI_SSM = ("/aws/service/deeplearning/ami/x86_64/"
             "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id")
TEMPLATE = pathlib.Path(__file__).parent / "cfn" / "gpu-lab.yaml"
NIM_IMAGE = "nvcr.io/nim/meta/llama-3.1-8b-instruct:2.0.13"   # pinned; the NIM docs' tag
NGC_KEY_PARAM = "/ncp-aai/ngc-api-key"    # SecureString you create with `ngc-key`
NIM_PORT = 8000


def client(name):
    return boto3.client(name, region_name=REGION)


def use_region(name):
    global REGION
    REGION = name


def find_stack_region():
    """Point REGION at the region that holds the lab stack; return it, or None."""
    for r in REGIONS:
        use_region(r)
        if _stack_exists(client("cloudformation")):
            return r
    use_region(REGIONS[0])
    return None


def gpu_quota():
    q = client("service-quotas").get_service_quota(ServiceCode="ec2", QuotaCode=QUOTA_CODE)
    return q["Quota"]["Value"]


def cmd_quota(args):
    value = gpu_quota()
    print(f"G and VT On-Demand quota in {REGION}: {value:g} vCPUs "
          f"(this course needs {INSTANCE_VCPUS})")
    if args.request:
        r = client("service-quotas").request_service_quota_increase(
            ServiceCode="ec2", QuotaCode=QUOTA_CODE, DesiredValue=float(args.request))
        print(f"Requested {args.request} vCPUs. Request status: "
              f"{r['RequestedQuota']['Status']}. AWS reviews these; it can take a while.")
    elif value < INSTANCE_VCPUS:
        print(f"Too low. Run: python setup/aws_lab.py quota --request {INSTANCE_VCPUS * 2}")
        return 1
    return 0


def cmd_budget(args):
    account = client("sts").get_caller_identity()["Account"]
    sub = [{"SubscriptionType": "EMAIL", "Address": args.email}]
    notes = [
        {"Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN",
                          "Threshold": 80, "ThresholdType": "PERCENTAGE"}, "Subscribers": sub},
        {"Notification": {"NotificationType": "FORECASTED", "ComparisonOperator": "GREATER_THAN",
                          "Threshold": 100, "ThresholdType": "PERCENTAGE"}, "Subscribers": sub},
    ]
    try:
        client("budgets").create_budget(
            AccountId=account,
            Budget={"BudgetName": BUDGET_NAME, "BudgetType": "COST", "TimeUnit": "MONTHLY",
                    "BudgetLimit": {"Amount": str(args.limit), "Unit": "USD"}},
            NotificationsWithSubscribers=notes)
        print(f"Budget '{BUDGET_NAME}' created: {args.limit} USD/month, emails to {args.email} "
              "at 80% of actual spend and 100% of forecast.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "DuplicateRecordException":
            print(f"Budget '{BUDGET_NAME}' already exists.")
        else:
            raise
    return 0


def stack_instance():
    cf = client("cloudformation")
    try:
        outs = cf.describe_stacks(StackName=STACK)["Stacks"][0].get("Outputs", [])
    except ClientError:
        return None
    return next((o["OutputValue"] for o in outs if o["OutputKey"] == "InstanceId"), None)


def gpu_subnets(only_az=None):
    """[(zone, default subnet)] for the zones in REGION that offer INSTANCE_TYPE."""
    ec2 = client("ec2")
    offered = {o["Location"] for o in ec2.describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": [INSTANCE_TYPE]}])["InstanceTypeOfferings"]}
    subnets = ec2.describe_subnets(Filters=[{"Name": "default-for-az", "Values": ["true"]}])["Subnets"]
    pairs = sorted((s["AvailabilityZone"], s["SubnetId"]) for s in subnets
                   if s["AvailabilityZone"] in offered)
    return [p for p in pairs if not only_az or p[0] == only_az]


def console_checks(instance_id):
    """Return the NCPAAI| lines the instance printed to its console, as a dict."""
    out = client("ec2").get_console_output(InstanceId=instance_id, Latest=True).get("Output", "")
    found = {}
    for line in out.splitlines():
        if "NCPAAI| " in line:
            body = line.split("NCPAAI| ", 1)[1].strip()
            key, _, val = body.partition("=")
            found[key] = val
    return found


def _up_in_region(args):
    """Create the stack in REGION, trying each zone. Returns "ok", "next" (no capacity/quota) or "error"."""
    quota = gpu_quota()
    if quota < INSTANCE_VCPUS:
        print(f"{REGION}: G and VT quota is {quota:g} vCPUs (need {INSTANCE_VCPUS}); skipping. "
              f"To use it: python setup/aws_lab.py quota --region {REGION} --request {INSTANCE_VCPUS * 2}")
        return "next"
    ami = client("ssm").get_parameter(Name=DLAMI_SSM)["Parameter"]["Value"]
    image = client("ec2").describe_images(ImageIds=[ami])["Images"][0]
    name, root_dev = image["Name"], image["RootDeviceName"]
    print(f"{REGION}: AMI {ami} ({name})")
    if args.lab == "m05":
        print(f"Module 5: the instance also pulls and starts {NIM_IMAGE} on port {NIM_PORT}.")
    cf = client("cloudformation")
    params = [{"ParameterKey": "ImageId", "ParameterValue": ami},
              {"ParameterKey": "InstanceType", "ParameterValue": INSTANCE_TYPE},
              {"ParameterKey": "MaxMinutes", "ParameterValue": str(MAX_MINUTES)},
              {"ParameterKey": "Lab", "ParameterValue": args.lab},
              {"ParameterKey": "NimImage", "ParameterValue": NIM_IMAGE},
              {"ParameterKey": "RootDeviceName", "ParameterValue": root_dev},
              {"ParameterKey": "NgcKeyParameter", "ParameterValue": NGC_KEY_PARAM}]
    # GPU capacity differs between Availability Zones. Try each zone that offers the
    # instance type (default subnets only); on "Insufficient capacity", delete and move on.
    zones = gpu_subnets(args.az)
    if not zones:
        print(f"{REGION}: no default subnet in a zone that offers {INSTANCE_TYPE}; skipping.")
        return "next"
    for n, (az, subnet) in enumerate(zones, 1):
        started = dt.datetime.now()
        print(f"[{n}/{len(zones)}] Creating stack {STACK} in {az} ... (the instance terminates itself "
              f"about {(started + dt.timedelta(minutes=MAX_MINUTES)):%H:%M} at the latest)")
        cf.create_stack(
            StackName=STACK, TemplateBody=TEMPLATE.read_text(),
            Parameters=params + [{"ParameterKey": "SubnetId", "ParameterValue": subnet}],
            Capabilities=["CAPABILITY_IAM"],      # m05 adds an instance role for Session Manager
            Tags=[{"Key": "Project", "Value": "ncp-aai-labs"}])
        try:
            cf.get_waiter("stack_create_complete").wait(StackName=STACK)
            break
        except Exception:
            ev = cf.describe_stack_events(StackName=STACK)["StackEvents"]
            fails = [e.get("ResourceStatusReason", "") for e in ev
                     if e["ResourceStatus"] == "CREATE_FAILED"]
            why = fails[-1] if fails else "unknown"          # the first failure, not the cascade
            print(f"   failed: {why[:200]}")
            if "capacity" not in why.lower():
                print("Clean up with: python setup/aws_lab.py down")
                return "error"
            cf.delete_stack(StackName=STACK)
            cf.get_waiter("stack_delete_complete").wait(StackName=STACK)
    else:
        print(f"{REGION}: no {INSTANCE_TYPE} capacity in any zone.")
        return "next"
    return "ok"


def cmd_up(args):
    if find_stack_region():
        print(f"Stack {STACK} already exists in {REGION}. Run `python setup/aws_lab.py down` first.")
        return 1
    regions = [args.region] if args.region else REGIONS
    for region in regions:
        use_region(region)
        result = _up_in_region(args)
        if result == "ok":
            break
        if result == "error":
            return 1
    else:
        print(f"No {INSTANCE_TYPE} capacity (or quota) in {', '.join(regions)} right now. "
              "Nothing is left running; try again later.")
        return 1
    iid = stack_instance()
    print(f"Instance {iid} ({INSTANCE_TYPE}) is running. Waiting for its checks "
          "(console output can take several minutes to appear)...")
    shown = set()
    for _ in range(100 if args.lab == "m05" else 40):   # up to ~50 / ~20 minutes
        checks = console_checks(iid)
        for k, v in checks.items():                      # print NIM progress as it arrives
            if k.startswith(("ngc_", "nim_", "profiles")) and (k, v) not in shown:
                shown.add((k, v))
                print(f"  {k:18} {v}")
        if "DONE" in checks:
            break
        time.sleep(30)
    else:
        print("No check results yet. Run `python setup/check_env.py --mode aws` in a few "
              "minutes, and `python setup/aws_lab.py down` when you are finished.")
        return 1
    for k, v in checks.items():
        if k not in ("START", "DONE") and not k.startswith(("ngc_", "nim_", "profiles")):
            print(f"  {k:18} {v}")
    if args.lab == "m05":
        if checks.get("nim_ready", "").startswith("yes"):
            print("NIM is ready. Next, in a second terminal: python setup/aws_lab.py tunnel\n"
                  "then: python m05/check.py --aws   and finally: python setup/aws_lab.py down")
        else:
            print("The NIM did not become ready (see the lines above). "
                  "Delete everything with: python setup/aws_lab.py down")
            return 1
        return 0
    print("Next: python setup/check_env.py --mode aws, then python setup/aws_lab.py down")
    return 0


def cmd_status(args):
    find_stack_region()
    iid = stack_instance()
    if not iid:
        print(f"No stack named {STACK}: nothing is running from this course.")
        return 0
    inst = client("ec2").describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
    state, launched = inst["State"]["Name"], inst["LaunchTime"]
    print(f"Stack {STACK}: instance {iid} is {state}.")
    if state == "running":
        left = MAX_MINUTES - (dt.datetime.now(dt.timezone.utc) - launched).total_seconds() / 60
        print(f"It terminates itself in about {max(left, 0):.0f} minutes.")
    print("Delete everything with: python setup/aws_lab.py down")
    return 0


def cmd_down(args):
    find_stack_region()
    cf = client("cloudformation")
    if not stack_instance() and not _stack_exists(cf):
        print("Nothing to delete.")
        return 0
    cf.delete_stack(StackName=STACK)
    print(f"Deleting stack {STACK} (instance, disk, launch template, security group)...")
    cf.get_waiter("stack_delete_complete").wait(StackName=STACK)
    left = client("ec2").describe_instances(Filters=[
        {"Name": "tag:Project", "Values": ["ncp-aai-labs"]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}])
    if any(r["Instances"] for r in left["Reservations"]):
        print("WARNING: an ncp-aai-labs instance still exists. Check the EC2 console.")
        return 1
    print("Deleted. No course instances are left in", REGION)
    return 0


def cmd_ngc_key(args):
    """Store the NGC API key as an SSM SecureString. The key is typed hidden and never printed."""
    import getpass
    key = getpass.getpass("NGC API key (input hidden): ").strip()
    if not key:
        print("No key entered; nothing stored.")
        return 1
    for r in ([args.region] if args.region else REGIONS):     # the instance reads it in its own region
        use_region(r)
        client("ssm").put_parameter(Name=NGC_KEY_PARAM, Value=key, Type="SecureString", Overwrite=True)
        print(f"Stored as SecureString {NGC_KEY_PARAM} in {r}.")
    print("Only the lab instance's role (and you) can read it.")
    return 0


def cmd_tunnel(args):
    """Forward localhost:8000 on this machine to port 8000 on the lab instance (Session Manager)."""
    import shutil
    import subprocess
    find_stack_region()
    iid = stack_instance()
    if not iid:
        print("No lab instance. Run `python setup/aws_lab.py up --lab m05` first.")
        return 1
    if not shutil.which("aws") or not shutil.which("session-manager-plugin"):
        print("Needs the AWS CLI and the Session Manager plugin on this machine (see setup/README.md).")
        return 1
    print(f"Forwarding http://localhost:{args.port} -> {iid}:{NIM_PORT}. Leave this running; Ctrl+C to stop.")
    return subprocess.call([
        "aws", "ssm", "start-session", "--region", REGION, "--target", iid,
        "--document-name", "AWS-StartPortForwardingSession",
        "--parameters", f'{{"portNumber":["{NIM_PORT}"],"localPortNumber":["{args.port}"]}}'])


def _stack_exists(cf):
    try:
        cf.describe_stacks(StackName=STACK)
        return True
    except ClientError:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("quota"); q.add_argument("--request", type=int)
    b = sp.add_parser("budget"); b.add_argument("--email", required=True)
    b.add_argument("--limit", type=int, default=10, help="USD per month (default 10)")
    u = sp.add_parser("up")
    u.add_argument("--lab", choices=["none", "m05"], default="none",
                   help="m05 also starts a Llama 3.1 8B NIM (Module 5)")
    u.add_argument("--az", help="only try this Availability Zone, e.g. us-east-1b")
    t = sp.add_parser("tunnel"); t.add_argument("--port", type=int, default=NIM_PORT)
    sp.add_parser("ngc-key"); sp.add_parser("status"); sp.add_parser("down")
    p.add_argument("--region", choices=REGIONS,
                   help=f"one region only (default: {REGIONS[0]}; `up` and `ngc-key` use all of {REGIONS})")
    args = p.parse_args()
    if args.region:
        use_region(args.region)
    return {"quota": cmd_quota, "budget": cmd_budget, "up": cmd_up, "ngc-key": cmd_ngc_key,
            "tunnel": cmd_tunnel, "status": cmd_status, "down": cmd_down}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
