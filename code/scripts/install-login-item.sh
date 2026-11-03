#!/usr/bin/env bash
# 用「登录项」方式实现开机自启（替代 launchd LaunchAgent）。
#
# 为什么不用 launchd：macOS 的 TCC 保护 ~/Documents，launchd 启动的任务
# 读不到项目里的代码/日志，任务初始化就失败（exit 78 EX_CONFIG）。
# 登录项由 Terminal 打开 .command 执行，而 Terminal 已经被授予了 Documents 权限。
#
# 用法：
#   bash code/scripts/install-login-item.sh          # 安装 + 立即启动
#   bash code/scripts/install-login-item.sh --remove # 移除登录项
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$CODE_ROOT/.." && pwd)"
PY="$ROOT/.venv-funasr/bin/python"
PORT="${PORT:-8000}"
SUPPORT="$HOME/Library/Application Support/TingJi"
CMD="$SUPPORT/TingJi.command"
START="$SUPPORT/start.sh"
LOG_DIR="$ROOT/output/.runtime/logs"
LABEL="com.tingji.service"
UID_=$(id -u)

remove_login_item() {
  osascript -e "tell application \"System Events\" to delete (every login item whose path is \"$CMD\")" 2>/dev/null || true
  rm -rf "$SUPPORT"
  echo "✅ 已移除登录项与启动脚本"
}

if [ "${1:-}" = "--remove" ]; then
  remove_login_item
  launchctl bootout "gui/$UID_/$LABEL" 2>/dev/null || true
  rm -f "$HOME/Library/LaunchAgents/$LABEL.plist"
  pkill -f "app.cli .*service" 2>/dev/null || true
  echo "✅ 服务已停止，launchd 残留已清理"
  exit 0
fi

[ -x "$PY" ] || { echo "❌ 找不到虚拟环境：$PY"; exit 1; }
mkdir -p "$SUPPORT" "$LOG_DIR"

cat > "$START" <<EOF
#!/bin/bash
# 由登录项调用：后台启动「听记」，已在运行则跳过。
ROOT="$ROOT"
CODE_ROOT="$CODE_ROOT"
PY="$PY"
PORT="$PORT"
LOG_DIR="$LOG_DIR"

if curl -sf "http://127.0.0.1:\$PORT/api/health" >/dev/null 2>&1; then
  echo "已在运行，跳过启动"
  exit 0
fi
mkdir -p "\$LOG_DIR"
cd "\$CODE_ROOT"
nohup "\$PY" -m app.cli -c config/config.yaml service --port "\$PORT" \\
  >> "\$LOG_DIR/service.log" 2>&1 &
echo \$! > "\$LOG_DIR/service.pid"
echo "启动中，pid=\$(cat "\$LOG_DIR/service.pid")"
EOF
chmod +x "$START"

cat > "$CMD" <<EOF
#!/bin/bash
# 登录项入口：调用 start.sh 后立刻退出，不占用 Terminal 窗口。
"$START"
exit 0
EOF
chmod +x "$CMD"

echo "① 启动脚本"
echo "   → $START"
echo "② 登录项入口"
echo "   → $CMD"

# 清理旧的 launchd 任务，避免两套机制打架
launchctl bootout "gui/$UID_/$LABEL" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist" 2>/dev/null || true
echo "③ 已清理旧的 launchd 任务"

# 注册登录项（首次会请求「控制 System Events」权限，请点允许）
echo "④ 注册登录项"
if osascript -e "tell application \"System Events\" to make login item at end with properties {path:\"$CMD\", hidden:true}" >/dev/null 2>/tmp/tingji-login.err; then
  echo "   ✅ 已加入登录项"
else
  echo "   ⚠️ 自动添加失败：$(cat /tmp/tingji-login.err)"
  echo "      请手动添加：系统设置 → 通用 → 登录项 → 打开时 → + → 选择上面的 .command 文件"
fi

echo "⑤ 立即启动服务"
pkill -f "app.cli .*service" 2>/dev/null || true
sleep 1
bash "$START"
for i in $(seq 1 25); do
  if curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
    echo "   ✅ $(curl -s "http://127.0.0.1:$PORT/api/health")"
    echo "   🌐 http://127.0.0.1:$PORT/"
    echo "   📄 日志：$LOG_DIR/service.log"
    echo
    echo "下次开机登录后会自动启动（登录项：${CMD}）"
    exit 0
  fi
  sleep 1
done
echo "   ❌ 启动失败，最近日志："
tail -20 "$LOG_DIR/service.log" 2>/dev/null || true
exit 1
