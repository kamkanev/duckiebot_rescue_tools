import os
import sys
import subprocess


def main():
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    graph_dir = os.path.join(repo_root, "graph_drawer")
    main_py = os.path.join(graph_dir, "main.py")

    image_arg = sys.argv[1] if len(sys.argv) > 1 else "copy2.jpg"
    # Resolve to absolute path under graph_drawer/assets if needed
    if not os.path.isabs(image_arg) and not image_arg.startswith("assets" + os.sep):
        image_path = os.path.join(graph_dir, "assets", image_arg)
    else:
        image_path = image_arg if os.path.isabs(image_arg) else os.path.join(graph_dir, image_arg)

    cmd = [sys.executable, main_py, image_path]
    subprocess.run(cmd, check=False, cwd=graph_dir)


if __name__ == "__main__":
    main()
