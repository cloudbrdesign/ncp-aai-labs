"""AWS helper for the NCP-AAI course labs (AWS-GPU mode).

    python setup/aws_lab.py quota                 # show your GPU instance quota
    python setup/aws_lab.py quota --request 8     # ask AWS to raise it to 8 vCPUs
    python setup/aws_lab.py budget --email you@example.com [--limit 10]
    python setup/aws_lab.py up                    # launch one GPU instance, run checks
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


def client(name):
    return boto3.client(name, region_name=REGION)


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


def cmd_up(args):
    if gpu_quota() < INSTANCE_VCPUS:
        print("Your GPU quota is too low. Run `python setup/aws_lab.py quota` first.")
        return 1
    ami = client("ssm").get_parameter(Name=DLAMI_SSM)["Parameter"]["Value"]
    name = client("ec2").describe_images(ImageIds=[ami])["Images"][0]["Name"]
    print(f"AMI: {ami} ({name})")
    cf = client("cloudformation")
    if _stack_exists(cf):
        print(f"Stack {STACK} already exists. Run `python setup/aws_lab.py down` first.")
        return 1
    cf.create_stack(
        StackName=STACK, TemplateBody=TEMPLATE.read_text(),
        Parameters=[{"ParameterKey": "ImageId", "ParameterValue": ami},
                    {"ParameterKey": "InstanceType", "ParameterValue": INSTANCE_TYPE},
                    {"ParameterKey": "MaxMinutes", "ParameterValue": str(MAX_MINUTES)}],
        Tags=[{"Key": "Project", "Value": "ncp-aai-labs"}])
    started = dt.datetime.now()
    print(f"Creating stack {STACK} ... (the instance terminates itself about "
          f"{(started + dt.timedelta(minutes=MAX_MINUTES)):%H:%M} at the latest)")
    try:
        cf.get_waiter("stack_create_complete").wait(StackName=STACK)
    except Exception:
        ev = cf.describe_stack_events(StackName=STACK)["StackEvents"]
        why = next((e.get("ResourceStatusReason") for e in ev
                    if e["ResourceStatus"].endswith("FAILED")), "unknown")
        print(f"Launch failed: {why}\nClean up with: python setup/aws_lab.py down")
        return 1
    iid = stack_instance()
    print(f"Instance {iid} ({INSTANCE_TYPE}) is running. Waiting for its checks "
          "(console output can take several minutes to appear)...")
    for _ in range(40):                     # up to ~20 minutes
        checks = console_checks(iid)
        if "DONE" in checks:
            break
        time.sleep(30)
    else:
        print("No check results yet. Run `python setup/check_env.py --mode aws` in a few "
              "minutes, and `python setup/aws_lab.py down` when you are finished.")
        return 1
    for k, v in checks.items():
        if k not in ("START", "DONE"):
            print(f"  {k:18} {v}")
    print("Next: python setup/check_env.py --mode aws, then python setup/aws_lab.py down")
    return 0


def cmd_status(args):
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
    sp.add_parser("up"); sp.add_parser("status"); sp.add_parser("down")
    args = p.parse_args()
    return {"quota": cmd_quota, "budget": cmd_budget, "up": cmd_up,
            "status": cmd_status, "down": cmd_down}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
