"""Repository-level ordering of source files (DeepSeek-Coder data recipe).

Instead of training on files in arbitrary order, DeepSeek-Coder parses the
dependencies between files of a repository and concatenates them so that a
file appears after the files it imports. This is a topological sort; cycles,
which happen in real projects, are broken by taking the file with the fewest
unresolved dependencies. See Chapter 17 of "Secrets of DeepSeek AI".

Run:
    python repo_order.py
"""
import re

IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.M)


def dependencies(files: dict[str, str]) -> dict[str, set[str]]:
    """Map each module to the set of *local* modules it imports."""
    local = {name.removesuffix(".py").replace("/", "."): name for name in files}
    deps = {}
    for name, text in files.items():
        found = set()
        for a, b in IMPORT_RE.findall(text):
            mod = a or b
            for cand in (mod, mod.split(".")[0]):
                if cand in local and local[cand] != name:
                    found.add(local[cand])
        deps[name] = found
    return deps


def repo_order(files: dict[str, str]) -> list[str]:
    deps = dependencies(files)              # file -> local imports
    remaining = {f: set(d) for f, d in deps.items()}
    order = []
    while remaining:
        ready = sorted(f for f, d in remaining.items() if not d)
        if not ready:                       # a cycle: break it
            ready = [min(remaining,
                         key=lambda f: (len(remaining[f]), f))]
        for f in ready:
            order.append(f)
            remaining.pop(f)
            for d in remaining.values():
                d.discard(f)
    return order


if __name__ == "__main__":
    repo = {
        "app.py": "from service import run\nimport config\nrun(config.PORT)\n",
        "service.py": "from db import connect\nfrom utils import log\ndef run(p): log(p)\n",
        "db.py": "import config\ndef connect(): return config.DSN\n",
        "utils.py": "def log(x): print(x)\n",
        "config.py": "PORT = 8080\nDSN = 'sqlite://'\n",
        "a.py": "import b\n",
        "b.py": "import a\n",
    }
    for f, d in sorted(dependencies(repo).items()):
        print(f"{f:11s} imports {sorted(d)}")
    order = repo_order(repo)
    print("\ntraining order:", " -> ".join(order))
    pos = {f: i for i, f in enumerate(order)}
    ok = all(pos[d] < pos[f] for f, ds in dependencies(repo).items()
             for d in ds if {f, d} != {"a.py", "b.py"})
    print("every acyclic dependency precedes its user:", ok)
