#!/usr/bin/env bash
# Start Prometheus (:9090) and Grafana (:3000) for the M8 lab, from Homebrew, with this folder's config.
#   brew install prometheus grafana            # once
#   bash m08/observability/run_stack.sh        # from the repo root; Ctrl-C stops both
# Grafana opens without a login (anonymous admin, for a lab on your own laptop only) on the
# "Support desk (M8)" dashboard. Data goes to m08/state/prometheus and m08/state/grafana.
set -euo pipefail
cd "$(dirname "$0")/../.."                      # the repo root
OBS=m08/observability
STATE=m08/state
mkdir -p "$STATE/prometheus" "$STATE/grafana" "$STATE/logs"
command -v prometheus >/dev/null || { echo "[ERROR] prometheus not found: brew install prometheus"; exit 1; }
command -v grafana >/dev/null || { echo "[ERROR] grafana not found: brew install grafana"; exit 1; }
# Grafana's home folder (it holds conf/defaults.ini and public/); Homebrew has put it in either place
if [ -z "${GRAFANA_HOME:-}" ]; then
  for d in "$(brew --prefix grafana)/libexec" "$(brew --prefix grafana)/share/grafana"; do
    [ -f "$d/conf/defaults.ini" ] && GRAFANA_HOME="$d" && break
  done
fi
[ -n "${GRAFANA_HOME:-}" ] || { echo "[ERROR] Grafana's home folder not found; set GRAFANA_HOME"; exit 1; }

prometheus --config.file="$OBS/prometheus.yml" --storage.tsdb.path="$STATE/prometheus" \
  --web.listen-address=127.0.0.1:9090 > "$STATE/logs/prometheus.log" 2>&1 &
PROM=$!

export GF_SERVER_HTTP_ADDR=127.0.0.1 GF_SERVER_HTTP_PORT=3000
export GF_PATHS_DATA="$PWD/$STATE/grafana" GF_PATHS_LOGS="$PWD/$STATE/logs"
export GF_PATHS_PROVISIONING="$PWD/$OBS/grafana/provisioning" M08_DASHBOARDS="$PWD/$OBS/grafana"
export GF_AUTH_ANONYMOUS_ENABLED=true GF_AUTH_ANONYMOUS_ORG_ROLE=Admin GF_AUTH_DISABLE_LOGIN_FORM=true
export GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH="$PWD/$OBS/grafana/support_desk.json"
grafana server --homepath "$GRAFANA_HOME" > "$STATE/logs/grafana.log" 2>&1 &
GRAF=$!

trap 'kill $PROM $GRAF 2>/dev/null; wait 2>/dev/null; echo "[stack] stopped"' INT TERM EXIT
echo "[stack] Prometheus http://127.0.0.1:9090 (alerts: /alerts)   Grafana http://127.0.0.1:3000"
echo "[stack] logs: $STATE/logs/prometheus.log, $STATE/logs/grafana.log   (Ctrl-C stops both)"
wait
