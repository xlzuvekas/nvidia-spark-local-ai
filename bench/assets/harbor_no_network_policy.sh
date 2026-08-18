#!/bin/sh
set -eu

NFTABLES_RULESET_NAME=gost_egress
RELAY_PORT=18080
RELAY_SENTINEL=sparkbench-relay.invalid

usage() {
  echo "Usage: network-policy show|allow-all|deny-all|rules" >&2
}

ensure_default_drop() {
  if nft list table inet "$NFTABLES_RULESET_NAME" >/dev/null 2>&1; then
    return
  fi
  nft --file - <<EOF
add table inet $NFTABLES_RULESET_NAME
add chain inet $NFTABLES_RULESET_NAME output { type filter hook output priority filter; policy drop; }
EOF
}

install_deny_all() {
  ensure_default_drop
  nft --file - <<EOF
flush chain inet $NFTABLES_RULESET_NAME output
EOF
}

install_agent_relay_only() {
  ensure_default_drop
  nft --file - <<EOF
flush chain inet $NFTABLES_RULESET_NAME output
add rule inet $NFTABLES_RULESET_NAME output oifname "lo" meta nfproto ipv4 ip daddr 127.0.0.1 tcp dport $RELAY_PORT accept
add rule inet $NFTABLES_RULESET_NAME output oifname "lo" meta nfproto ipv4 ip daddr 127.0.0.1 tcp sport $RELAY_PORT ct state established accept
EOF
}

case "${1:-}" in
  show)
    if nft list table inet "$NFTABLES_RULESET_NAME" >/dev/null 2>&1; then
      echo "mode: controlled"
    else
      echo "mode: missing"
    fi
    ;;
  allow-all)
    echo "public network mode is disabled by the SparkBench campaign policy" >&2
    exit 2
    ;;
  deny-all)
    install_deny_all
    ;;
  rules)
    nft list table inet "$NFTABLES_RULESET_NAME"
    ;;
  allow)
    if [ "$#" -ne 2 ] || [ "$2" != "$RELAY_SENTINEL" ]; then
      echo "only the pinned SparkBench relay sentinel is accepted" >&2
      exit 2
    fi
    install_agent_relay_only
    ;;
  *)
    usage
    exit 2
    ;;
esac
