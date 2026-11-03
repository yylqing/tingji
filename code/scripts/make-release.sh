#!/usr/bin/env bash
# 打包源码发布物（GitHub Release 附件）。
#
# 用法：bash code/scripts/make-release.sh [tag]
# 产物：dist/tingji-<ver>.tar.gz、dist/tingji-<ver>.zip、dist/SHA256SUMS、dist/RELEASE_NOTES.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TAG="${1:-$(git describe --tags --abbrev=0 2>/dev/null || echo v0.0.0)}"
VER="${TAG#v}"
NAME="tingji-${VER}"
DIST="$ROOT/dist"
mkdir -p "$DIST"

echo "① 校验工作区"
if [ -n "$(git status --porcelain)" ]; then
  echo "   ⚠️ 有未提交改动，发布物只包含已提交内容"
fi

echo "② 生成源码包（git archive，自动排除 .gitignore 里的运行时目录）"
git archive --format=tar.gz --prefix="${NAME}/" -o "$DIST/${NAME}.tar.gz" "$TAG"
git archive --format=zip    --prefix="${NAME}/" -o "$DIST/${NAME}.zip"    "$TAG"

echo "③ 生成校验和"
( cd "$DIST" && shasum -a 256 "${NAME}.tar.gz" "${NAME}.zip" > SHA256SUMS )

echo "④ 生成发布说明"
NOTES="$DIST/RELEASE_NOTES.md"
{
  echo "# ${NAME}"
  echo
  echo "本次发布由 \`git tag ${TAG}\` 生成。"
  echo
  echo "## 安装"
  echo '```bash'
  echo "tar xzf ${NAME}.tar.gz && cd ${NAME}"
  echo "uv venv .venv-funasr --python 3.12"
  echo ".venv-funasr/bin/python -m pip install -r code/requirements.txt"
  echo "cp code/config/config.example.yaml code/config/config.yaml"
  echo "PYTHONPATH=code .venv-funasr/bin/python -m app.cli service --port 8000"
  echo '```'
  echo
  echo "## 文档"
  echo "- [使用手册](https://github.com/yylqing/tingji/blob/${TAG}/docs/10-使用手册.md)"
  echo "- [API 文档](https://github.com/yylqing/tingji/blob/${TAG}/docs/11-API文档.md)"
  echo "- [变更日志](https://github.com/yylqing/tingji/blob/${TAG}/CHANGELOG.md)"
  echo
  echo "## 校验"
  echo '```'
  cat "$DIST/SHA256SUMS"
  echo '```'
} > "$NOTES"

echo
echo "✅ 产物："
ls -lh "$DIST" | sed 's/^/   /'
echo
echo "下一步（二选一）："
echo "  A. 不能访问 GitHub API 时，网页创建 Release："
echo "     https://github.com/yylqing/tingji/releases/new?tag=${TAG}"
echo "     标题填 ${NAME}，正文粘贴 dist/RELEASE_NOTES.md，上传 dist/ 下两个包"
echo "  B. 已安装 gh 并登录："
echo "     gh release create ${TAG} ${DIST}/${NAME}.tar.gz ${DIST}/${NAME}.zip \\"
echo "       --title '${NAME}' --notes-file ${DIST}/RELEASE_NOTES.md"
