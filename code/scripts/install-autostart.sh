#!/usr/bin/env bash
# 安装/重装「听记」的 macOS 开机自启（LaunchAgent）。
# 目录约定：项目根 = code/ 的上一级；venv 与 output 在项目根，配置在 code/config。
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$CODE_ROOT/.." && pwd)"
PY="$ROOT/.venv-funasr/bin/python"
LABEL="com.tingji.service"
PORT="${PORT:-8000}"
PLIST_SRC="$CODE_ROOT/config/$LABEL.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/TingJi"   # launchd 日志（放 Documents 下会被 TCC 拦）
APP_LOG_DIR="$ROOT/output/.runtime/logs"

if [ ! -x "$PY" ]; then
  echo "❌ 找不到虚拟环境：$PY"
  exit 1
fi
mkdir -p "$LOG_DIR" "$APP_LOG_DIR"

echo "① 准备"
UID_=$(id -u)

# 逐级降级：某些 macOS 版本对 Documents 下的 LaunchAgent 会在 spawn 阶段报
# 78 EX_CONFIG（连进程都没起来）。依次尝试 full → minimal → no_workdir，
# 哪个能起来就用哪个，并把结果写回 config/ 里的 plist 供下次复用。
try_variant() {
  variant="$1"
  PORT="$PORT" VARIANT="$variant" PYTHONPATH="$CODE_ROOT" "$PY" - "$CODE_ROOT" "$PLIST_SRC" <<'PYEOF'
import os, pathlib, plistlib, sys
from app import service as svc
code_root = pathlib.Path(sys.argv[1]).resolve()
root = code_root.parent
pl = svc.build_plist(
    python=str(root / ".venv-funasr/bin/python"),
    root=code_root,
    config=str(code_root / "config/config.yaml"),
    log_dir=pathlib.Path.home() / "Library" / "Logs" / "TingJi",
    port=int(os.environ["PORT"]),
    variant=os.environ["VARIANT"],
)
pathlib.Path(sys.argv[2]).write_bytes(plistlib.dumps(pl))
PYEOF
  cp "$PLIST_SRC" "$PLIST_DST"

  # 命令预检（路径写错会在这里暴露）
  "$PY" - "$PLIST_SRC" "$CODE_ROOT" <<'PYEOF' || return 1
import os, pathlib, plistlib, subprocess, sys
pl = plistlib.loads(pathlib.Path(sys.argv[1]).read_bytes())
args = pl["ProgramArguments"]; cwd = pl.get("WorkingDirectory") or sys.argv[2]
env = {**os.environ, **pl.get("EnvironmentVariables", {})}
r = subprocess.run(args + ["--help"], cwd=cwd, capture_output=True, text=True, timeout=20, env=env)
if r.returncode != 0:
    print("   ❌ 预检失败：", (r.stderr or r.stdout).strip()[:200]); sys.exit(1)
PYEOF

  launchctl bootout "gui/$UID_/$LABEL" 2>/dev/null || true
  launchctl bootout "gui/$UID_" "$PLIST_DST" 2>/dev/null || true
  for i in 1 2 3; do launchctl print "gui/$UID_/$LABEL" >/dev/null 2>&1 || break; sleep 1; done
  launchctl bootstrap "gui/$UID_" "$PLIST_DST" 2>/tmp/tingji-bootstrap.err || {
    echo "   [$variant] bootstrap 失败：$(cat /tmp/tingji-bootstrap.err)"; return 1; }
  launchctl enable "gui/$UID_/$LABEL" 2>/dev/null || true
  launchctl kickstart -k "gui/$UID_/$LABEL" 2>/dev/null || true

  for i in $(seq 1 8); do
    if curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  echo "   [$variant] 未起来：$(launchctl print "gui/$UID_/$LABEL" 2>/dev/null | grep -E 'last exit' | head -1)"
  return 1
}

# 端口被手工实例占用时先停掉，避免 launchd 实例绑定失败、健康检查误报
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "   端口 $PORT 被占用，先停止旧实例"
  pkill -f "app.cli .*service" 2>/dev/null || true
  sleep 2
fi

echo "② 安装到 $PLIST_DST 并启动"
mkdir -p "$HOME/Library/LaunchAgents"
OK_VARIANT=""
for v in full minimal no_workdir; do
  echo "   尝试 plist 变体：$v"
  if try_variant "$v"; then OK_VARIANT="$v"; break; fi
done

if [ -z "$OK_VARIANT" ]; then
  echo "❌ 三种 plist 变体都没能启动"
  echo "   —— launchctl 状态 ——"
  launchctl print "gui/$UID_/$LABEL" 2>&1 | sed -n '1,18p'
  echo "   —— 系统日志（launchd）——"
  log show --last 3m --style compact --predicate 'process == "launchd"' 2>/dev/null | grep -i "$LABEL" | tail -5 || true

  echo "   —— launchd 自检 ——"
  selftest() {
    name="$1"; shift
    cat > "/tmp/tingji-$name.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>com.tingji.$name</string>
<key>ProgramArguments</key><array>$(for a in "$@"; do printf '<string>%s</string>' "$a"; done)</array>
<key>RunAtLoad</key><true/>
<key>StandardOutPath</key><string>/tmp/tingji-$name.out</string>
<key>StandardErrorPath</key><string>/tmp/tingji-$name.err</string>
</dict></plist>
PLIST
    launchctl bootout "gui/$UID_/com.tingji.$name" 2>/dev/null || true
    if launchctl bootstrap "gui/$UID_" "/tmp/tingji-$name.plist" 2>/tmp/tingji-$name.boot; then
      sleep 1
      st=$(launchctl print "gui/$UID_/com.tingji.$name" 2>/dev/null | grep -E "last exit|state" | head -2 | tr '\n' ' ')
      echo "      $name: $st"
    else
      echo "      $name: bootstrap 失败 $(cat /tmp/tingji-$name.boot)"
    fi
    launchctl bootout "gui/$UID_/com.tingji.$name" 2>/dev/null || true
  }
  selftest hello /bin/echo launchd-ok
  selftest python "$ROOT/.venv-funasr/bin/python" -c "print(1)"
  echo "   —— 结论 ——"
  if grep -q launchd-ok /tmp/tingji-hello.out 2>/dev/null; then
    echo "      launchd 能跑系统二进制；如果 python 自检失败，说明 Documents 目录被 TCC 限制"
    echo "      解决：系统设置 → 隐私与安全性 → 完全磁盘访问权限，把 $ROOT/.venv-funasr/bin/python 加进去"
  else
    echo "      launchd 连 /bin/echo 都跑不了：可能是系统禁止后台项，去 系统设置 → 通用 → 登录项 检查"
  fi
  exit 1
fi
echo "   ✅ 使用变体：$OK_VARIANT"

echo "③ 安装到 $PLIST_DST"
mkdir -p "$HOME/Library/LaunchAgents"
cp "$PLIST_SRC" "$PLIST_DST"

echo "④ 重新加载 launchd"
UID_=$(id -u)
# 旧任务必须真正卸载，否则 bootstrap 会因"已加载"失败、launchd 继续跑旧配置
launchctl bootout "gui/$UID_/$LABEL" 2>/dev/null || true
launchctl bootout "gui/$UID_" "$PLIST_DST" 2>/dev/null || true
for i in 1 2 3; do
  launchctl print "gui/$UID_/$LABEL" >/dev/null 2>&1 || break
  sleep 1
done
if launchctl print "gui/$UID_/$LABEL" >/dev/null 2>&1; then
  echo "   ⚠️ 旧任务仍在，等待 3 秒后重试"
  launchctl bootout "gui/$UID_/$LABEL" 2>/dev/null || true
  sleep 3
fi
# 端口被手工实例占用时先停掉，否则 launchd 实例绑定失败
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "   端口 $PORT 被占用，先停止旧实例"
  pkill -f "app.cli .*service" 2>/dev/null || true
  sleep 2
fi
if ! launchctl bootstrap "gui/$UID_" "$PLIST_DST" 2>/tmp/tingji-bootstrap.err; then
  echo "   ❌ bootstrap 失败：$(cat /tmp/tingji-bootstrap.err)"
  exit 1
fi
launchctl enable "gui/$UID_/$LABEL" 2>/dev/null || true
launchctl kickstart -k "gui/$UID_/$LABEL" 2>/dev/null || true

echo "③ 状态"
launchctl print "gui/$UID_/$LABEL" 2>&1 | grep -E "state|pid|last exit" | head -3 || true
echo "④ 健康检查"
echo "   ✅ $(curl -s "http://127.0.0.1:$PORT/api/health")"
echo "   🌐 http://127.0.0.1:$PORT/"
echo "   📄 自启日志：$LOG_DIR/service.log"
    echo "   📄 应用日志：$APP_LOG_DIR/service.log"
exit 0
