#!/usr/bin/env python3
"""Command-line wrapper for the authenticated Docker-to-loopback bridge."""

from bench.docker_bridge import main


if __name__ == "__main__":
    raise SystemExit(main())
