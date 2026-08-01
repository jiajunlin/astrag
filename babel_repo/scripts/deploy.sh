#!/usr/bin/env bash
set -euo pipefail

# Build the docker image and push it.
build_image() {
  docker build -t shop:latest .
  docker push shop:latest
}

function rollback {
  kubectl rollout undo deploy/shop
}

build_image
