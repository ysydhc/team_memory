#!/bin/bash
# Wait for daemon :3901/status to return 200 / 等待 daemon 就绪
for i in $(seq 1 30); do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:3901/status 2>/dev/null | grep -q 200; then
        echo "Daemon ready"
        exit 0
    fi
    sleep 1
done
echo "Daemon NOT ready after 30s"
exit 1
