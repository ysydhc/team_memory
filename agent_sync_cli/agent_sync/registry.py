import os
import hashlib


def save_to_local_registry(registry_dir: str, name: str, content: str) -> str:
    content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]

    # Check if any file with this hash already exists
    if os.path.exists(registry_dir):
        for existing_file in os.listdir(registry_dir):
            if existing_file.endswith(f"-{content_hash}.md"):
                return f"file://{os.path.join(registry_dir, existing_file)}"

    filename = f"{name}-{content_hash}.md"
    filepath = os.path.join(registry_dir, filename)

    if not os.path.exists(registry_dir):
        os.makedirs(registry_dir)

    with open(filepath, "w") as f:
        f.write(content)

    return f"file://{filepath}"
