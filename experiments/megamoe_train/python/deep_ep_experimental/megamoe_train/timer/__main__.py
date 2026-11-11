# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026

"""``python -m deep_ep_experimental.megamoe_train.timer <command>``.

Commands keep the training operator's own tool boundaries:

    forward      parse_timer_csv        raw -> parsed CSV (forward slots)
    backward     parse_backward_timer   raw -> parsed CSV + stage tables
    trace        trace_parser           parsed CSV -> Perfetto trace JSON
    check-schema compare protocol.py against the device headers

Examples:
    python -m deep_ep_experimental.megamoe_train.timer forward \\
        --dir <timer_dir> --rank-end 0
    python -m deep_ep_experimental.megamoe_train.timer backward \\
        --dir <timer_dir> --rank-end 0 --summary --r 32768 --k 7168 --f 8192
    python -m deep_ep_experimental.megamoe_train.timer trace \\
        --timer-dir <timer_dir> --rank-end 0 --phase backward
"""

import json
import sys

from . import parse_backward_timer, parse_timer_csv, protocol, trace_parser

USAGE = __doc__


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    command, rest = args[0], args[1:]
    if command == "forward":
        return parse_timer_csv.main(rest)
    if command == "backward":
        return parse_backward_timer.main(rest)
    if command == "trace":
        return trace_parser.main(rest)
    if command == "check-schema":
        print(
            json.dumps(
                {
                    "python": {"total_buffer_size": protocol.TOTAL_BUFFER_SIZE,
                               "bytes": protocol.timer_buffer_bytes()},
                    "cpp": protocol.check_cpp_schema(),
                }
            )
        )
        return 0
    raise SystemExit(f"unknown command {command!r}; expected forward|backward|trace|check-schema")


if __name__ == "__main__":
    raise SystemExit(main())
