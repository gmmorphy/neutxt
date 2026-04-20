"""Entry point for python -m neutxt <command>."""
import sys


COMMANDS = {
    "encode": "neutxt.encode",
    "play": "neutxt.play",
    "demo": "neutxt.demo",
    "llm": "neutxt.llm_demo",
}

USAGE = """Usage: python -m neutxt <command> [args...]

Commands:
  encode    Encode video/audio to binary .neutxt format
  play      Play a binary .neutxt file
  demo      Encode video to NEUTXT text format + decode to GIF
  llm       Send NEUTXT video through Claude API for manipulation

Examples:
  python -m neutxt demo input.mp4 --vq_ckpt ckpt.pt --vq_config cfg.yaml
  python -m neutxt llm input.mp4 --vq_ckpt ckpt.pt --vq_config cfg.yaml --task reverse
  python -m neutxt encode input.mp4 --vq_ckpt ckpt.pt --vq_config cfg.yaml
  python -m neutxt play output.neutxt --vq_ckpt ckpt.pt --vq_config cfg.yaml
"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print(USAGE)
        sys.exit(1)

    sys.argv = sys.argv[1:]  # shift so argparse in submodule sees correct args
    module = __import__(COMMANDS[cmd], fromlist=["main"])
    module.main()


if __name__ == "__main__":
    main()
