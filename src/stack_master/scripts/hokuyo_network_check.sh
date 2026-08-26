#!/usr/bin/env bash

# Read-only network guard for the Hokuyo UST-10LX.  This script intentionally
# never changes NetworkManager profiles, links, addresses, routes, or devices.

set -u

readonly INTERFACE="enP8p1s0"
readonly JETSON_CIDR="192.168.0.15/24"
readonly LIDAR_IP="192.168.0.10"
readonly CHECK_INTERVAL_SECONDS="5"
readonly PREFLIGHT_ATTEMPTS="5"

CHECK_LEVEL=""
CHECK_MESSAGE=""

log_info() {
  printf '[Hokuyo Network] %s\n' "$*"
}

log_warning() {
  printf '[Hokuyo Network] WARNING - %s\n' "$*" >&2
}

log_error() {
  printf '[Hokuyo Network] ERROR - %s\n' "$*" >&2
}

show_configuration() {
  log_info "Interface: ${INTERFACE}"
  log_info "Jetson IP: ${JETSON_CIDR%/*}"
  log_info "LiDAR IP: ${LIDAR_IP}"
}

check_network() {
  if [[ ! -d "/sys/class/net/${INTERFACE}" ]]; then
    CHECK_LEVEL="error"
    CHECK_MESSAGE="${INTERFACE} does not exist"
    return 1
  fi

  if ! ip -4 -o address show dev "${INTERFACE}" 2>/dev/null \
      | awk '{print $4}' \
      | grep -Fxq "${JETSON_CIDR}"; then
    CHECK_LEVEL="error"
    CHECK_MESSAGE="${INTERFACE} does not have ${JETSON_CIDR}"
    return 2
  fi

  if ! ping -I "${INTERFACE}" -c 1 -W 1 "${LIDAR_IP}" >/dev/null 2>&1; then
    CHECK_LEVEL="warning"
    CHECK_MESSAGE="LiDAR ${LIDAR_IP} unreachable via ${INTERFACE}"
    return 3
  fi

  CHECK_LEVEL=""
  CHECK_MESSAGE=""
  return 0
}

report_check_failure() {
  if [[ "${CHECK_LEVEL}" == "error" ]]; then
    log_error "${CHECK_MESSAGE}"
  else
    log_warning "${CHECK_MESSAGE}"
  fi
}

run_preflight() {
  local attempt

  show_configuration
  for ((attempt = 1; attempt <= PREFLIGHT_ATTEMPTS; attempt++)); do
    if check_network; then
      log_info "OK - LiDAR reachable"
      return 0
    fi
    report_check_failure

    if ((attempt < PREFLIGHT_ATTEMPTS)); then
      log_warning "Preflight attempt ${attempt}/${PREFLIGHT_ATTEMPTS} failed; retrying in 1 second"
      sleep 1
    fi
  done

  log_error "Preflight failed; urg_node will not be started"
  return 1
}

run_watchdog() {
  local previous_status=""
  local current_status

  show_configuration
  while true; do
    if check_network; then
      current_status="ok"
    else
      current_status="fault"
    fi

    if [[ "${current_status}" != "${previous_status}" ]]; then
      if [[ "${current_status}" == "ok" ]]; then
        log_info "OK - LiDAR reachable"
      else
        report_check_failure
        log_warning "Network fault detected; monitoring only (no interface or route changes)"
      fi
      previous_status="${current_status}"
    fi

    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

case "${1:-}" in
  --urg-node)
    shift
    if ! run_preflight; then
      exit 1
    fi
    exec ros2 run urg_node urg_node_driver "$@"
    ;;
  --preflight)
    shift
    if ! run_preflight; then
      exit 1
    fi
    if (($# == 0)); then
      exit 0
    fi
    exec "$@"
    ;;
  --watchdog)
    run_watchdog
    ;;
  *)
    printf 'Usage: %s --urg-node [ROS_ARGS ...] | --preflight [COMMAND ...] | --watchdog\n' "$0" >&2
    exit 2
    ;;
esac
