import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_CPP = ROOT / "3rdparty" / "llama.cpp" / "examples" / "server" / "server.cpp"
LLAMA_CPP = ROOT / "3rdparty" / "llama.cpp"
PATCHES = ROOT / "patches"

OLD_CORS_BLOCK = """    // CORS preflight
    svr->Options(R\"(.*)\", [](const httplib::Request &, httplib::Response & res) {
        // Access-Control-Allow-Origin is already set by middleware
        res.set_header(\"Access-Control-Allow-Credentials\", \"true\");
        res.set_header(\"Access-Control-Allow-Methods\",     \"POST\");
        res.set_header(\"Access-Control-Allow-Headers\",     \"*\");
        return res.set_content(\"\", \"text/html\"); // blank response, no data
    });
"""

NEW_CORS_BLOCK = """    // CORS preflight
    svr->Options(R\"(.*)\", [](const httplib::Request & req, httplib::Response & res) {
        // Access-Control-Allow-Origin is already set by middleware
        res.set_header(\"Access-Control-Allow-Credentials\", \"true\");
        res.set_header(\"Access-Control-Allow-Methods\",     \"GET, POST, OPTIONS\");

        const auto requested_headers = req.get_header_value(\"Access-Control-Request-Headers\");
        if (!requested_headers.empty()) {
            res.set_header(\"Access-Control-Allow-Headers\", requested_headers);
        } else {
            res.set_header(\"Access-Control-Allow-Headers\", \"*\");
        }

        return res.set_content(\"\", \"text/html\"); // blank response, no data
    });
"""


def ensure_server_cors_patch() -> None:
    if not SERVER_CPP.exists():
        print(f"Skipping llama.cpp CORS patch: file not found at {SERVER_CPP}")
        return

    content = SERVER_CPP.read_text(encoding="utf-8")
    if NEW_CORS_BLOCK in content:
        print("llama.cpp CORS patch already applied")
        return

    if OLD_CORS_BLOCK not in content:
        print("Failed to locate original CORS block in server.cpp", file=sys.stderr)
        sys.exit(1)

    SERVER_CPP.write_text(content.replace(OLD_CORS_BLOCK, NEW_CORS_BLOCK, 1), encoding="utf-8")
    print("Applied llama.cpp CORS patch")


def parse_hunk_header(line: str) -> int:
    old_range = line.split(" ", 2)[1]
    return int(old_range[1:].split(",", 1)[0])


def apply_unified_patch(patch: Path) -> bool:
    patch_lines = patch.read_text(encoding="utf-8").splitlines()
    i = 0
    applied_any = False

    while i < len(patch_lines):
        if not patch_lines[i].startswith("diff --git "):
            i += 1
            continue

        i += 1
        while i < len(patch_lines) and not patch_lines[i].startswith("--- "):
            i += 1
        if i >= len(patch_lines):
            return False

        i += 1
        if i >= len(patch_lines) or not patch_lines[i].startswith("+++ b/"):
            return False

        rel_path = patch_lines[i][6:]
        target = LLAMA_CPP / rel_path
        if not target.exists():
            print(f"Patch target not found: {target}", file=sys.stderr)
            return False

        original = target.read_text(encoding="utf-8")
        newline = "\r\n" if "\r\n" in original else "\n"
        content = original.splitlines()
        offset = 0

        i += 1
        while i < len(patch_lines):
            if patch_lines[i].startswith("diff --git "):
                break
            if not patch_lines[i].startswith("@@ "):
                i += 1
                continue

            start = parse_hunk_header(patch_lines[i]) - 1 + offset
            i += 1
            old_lines: list[str] = []
            new_lines: list[str] = []

            while i < len(patch_lines) and not patch_lines[i].startswith("@@ ") and not patch_lines[i].startswith("diff --git "):
                line = patch_lines[i]
                if line.startswith("\\ No newline"):
                    i += 1
                    continue

                marker = line[:1]
                value = line[1:]
                if marker == " ":
                    old_lines.append(value)
                    new_lines.append(value)
                elif marker == "-":
                    old_lines.append(value)
                elif marker == "+":
                    new_lines.append(value)
                else:
                    return False
                i += 1

            if content[start:start + len(old_lines)] != old_lines:
                return False

            content[start:start + len(old_lines)] = new_lines
            offset += len(new_lines) - len(old_lines)

        target.write_text(newline.join(content) + newline, encoding="utf-8")
        applied_any = True

    return applied_any


def apply_patch_file(patch: Path) -> None:
    if not patch.exists():
        print(f"Skipping patch: file not found at {patch}")
        return

    if apply_unified_patch(patch):
        print(f"Applied {patch.name}")
        return

    utils_hpp = LLAMA_CPP / "examples" / "server" / "utils.hpp"
    if utils_hpp.exists() and "tools_prompt(const json & body)" in utils_hpp.read_text(encoding="utf-8"):
        print(f"{patch.name} already applied")
        return

    print(f"Failed to apply {patch.name}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    ensure_server_cors_patch()
    apply_patch_file(PATCHES / "llama-server-tools.patch")


if __name__ == "__main__":
    main()
