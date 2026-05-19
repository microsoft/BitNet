import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
SERVER_CPP = ROOT / "3rdparty" / "llama.cpp" / "examples" / "server" / "server.cpp"
LLAMA_CPP = ROOT / "3rdparty" / "llama.cpp"
PATCHES = ROOT / "patches"

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


@dataclass
class PatchHunk:
    old_start: int
    old_lines: list[str]
    new_lines: list[str]


@dataclass
class FilePatch:
    target: Path
    hunks: list[PatchHunk]


def ensure_server_cors_patch() -> None:
    if not SERVER_CPP.exists():
        print(f"Skipping llama.cpp CORS patch: file not found at {SERVER_CPP}")
        return

    content = SERVER_CPP.read_text(encoding="utf-8")
    cors_comment = "    // CORS preflight"
    start = content.find(cors_comment)
    if start == -1:
        print("Failed to locate CORS preflight block in server.cpp", file=sys.stderr)
        sys.exit(1)

    end_marker = "    });"
    end = content.find(end_marker, start)
    if end == -1:
        print("Failed to locate end of CORS preflight block in server.cpp", file=sys.stderr)
        sys.exit(1)
    end += len(end_marker)

    current_block = content[start:end]
    if "Access-Control-Request-Headers" in current_block:
        print("llama.cpp CORS patch already applied")
        return

    required_markers = (
        "svr->Options",
        "httplib::Request &",
        "httplib::Response & res",
        'res.set_header("Access-Control-Allow-Methods"',
        'res.set_header("Access-Control-Allow-Headers"',
    )
    if not all(marker in current_block for marker in required_markers):
        print("Failed to locate expected CORS preflight lines in server.cpp", file=sys.stderr)
        sys.exit(1)

    newline = "\r\n" if "\r\n" in content else "\n"
    cors_block = NEW_CORS_BLOCK.rstrip("\n").replace("\n", newline)
    SERVER_CPP.write_text(content[:start] + cors_block + content[end:], encoding="utf-8")
    print("Applied llama.cpp CORS patch")


def parse_hunk_header(line: str) -> int:
    old_range = line.split(" ", 2)[1]
    return int(old_range[1:].split(",", 1)[0])


def parse_unified_patch(patch: Path) -> Optional[list[FilePatch]]:
    patch_lines = patch.read_text(encoding="utf-8").splitlines()
    i = 0
    file_patches: list[FilePatch] = []

    while i < len(patch_lines):
        if not patch_lines[i].startswith("diff --git "):
            i += 1
            continue

        i += 1
        while i < len(patch_lines) and not patch_lines[i].startswith("--- "):
            i += 1
        if i >= len(patch_lines):
            return None

        i += 1
        if i >= len(patch_lines) or not patch_lines[i].startswith("+++ b/"):
            return None

        rel_path = patch_lines[i][6:]
        target = LLAMA_CPP / rel_path
        if not target.exists():
            print(f"Patch target not found: {target}", file=sys.stderr)
            return None

        i += 1
        hunks: list[PatchHunk] = []
        while i < len(patch_lines):
            if patch_lines[i].startswith("diff --git "):
                break
            if not patch_lines[i].startswith("@@ "):
                i += 1
                continue

            old_start = parse_hunk_header(patch_lines[i])
            i += 1
            old_lines: list[str] = []
            new_lines: list[str] = []

            while i < len(patch_lines) and not patch_lines[i].startswith("@@ ") and not patch_lines[i].startswith("diff --git "):
                line = patch_lines[i]
                if line.startswith("\\ No newline"):
                    i += 1
                    continue
                if line == "":
                    old_lines.append("")
                    new_lines.append("")
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
                    return None
                i += 1

            hunks.append(PatchHunk(old_start=old_start, old_lines=old_lines, new_lines=new_lines))

        file_patches.append(FilePatch(target=target, hunks=hunks))

    return file_patches


def simulate_file_patch(file_patch: FilePatch) -> tuple[str, Optional[str]]:
    original = file_patch.target.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in original else "\n"
    has_trailing_newline = original.endswith(("\n", "\r"))
    original_lines = original.splitlines()
    patched_lines = original_lines.copy()
    apply_offset = 0
    already_offset = 0
    can_apply = True
    already_applied = True

    for hunk in file_patch.hunks:
        apply_start = hunk.old_start - 1 + apply_offset
        if patched_lines[apply_start:apply_start + len(hunk.old_lines)] == hunk.old_lines:
            patched_lines[apply_start:apply_start + len(hunk.old_lines)] = hunk.new_lines
            apply_offset += len(hunk.new_lines) - len(hunk.old_lines)
        else:
            can_apply = False

        already_start = hunk.old_start - 1 + already_offset
        if original_lines[already_start:already_start + len(hunk.new_lines)] != hunk.new_lines:
            already_applied = False
        already_offset += len(hunk.new_lines) - len(hunk.old_lines)

    if can_apply:
        patched_content = newline.join(patched_lines)
        if has_trailing_newline:
            patched_content += newline
        return "apply", patched_content
    if already_applied:
        return "already", None
    return "failed", None


def apply_unified_patch(patch: Path) -> str:
    file_patches = parse_unified_patch(patch)
    if not file_patches:
        return "failed"

    pending_writes: list[tuple[Path, str]] = []
    applied_any = False
    already_count = 0

    for file_patch in file_patches:
        status, new_content = simulate_file_patch(file_patch)
        if status == "failed":
            return "failed"
        if status == "already":
            already_count += 1
            continue
        pending_writes.append((file_patch.target, new_content or ""))
        applied_any = True

    if applied_any and already_count:
        print(f"Refusing to partially apply {patch.name}: some hunks are already present", file=sys.stderr)
        return "failed"

    for target, new_content in pending_writes:
        target.write_text(new_content, encoding="utf-8")

    return "applied" if applied_any else "already"


def apply_patch_file(patch: Path) -> None:
    if not patch.exists():
        print(f"Skipping patch: file not found at {patch}")
        return

    patch_status = apply_unified_patch(patch)
    if patch_status == "applied":
        print(f"Applied {patch.name}")
        return
    if patch_status == "already":
        print(f"{patch.name} already applied")
        return

    print(f"Failed to apply {patch.name}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    ensure_server_cors_patch()
    apply_patch_file(PATCHES / "llama-server-tools.patch")


if __name__ == "__main__":
    main()
