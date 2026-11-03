#!/usr/bin/env bash
# 生成基准测试音频：39.57s 中文"会议"，4 段发言、3 个可区分说话人，含英文术语 ClickHouse。
# 依赖：macOS `say`（Tingting / Meijia 两个中文音色可用）、ffmpeg。
#
# 注意：这是合成音频，只用于测吞吐与可行性，不能用于测准确率。
#   - 无背景噪声、无重叠说话
#   - 段间无停顿（直接拼接），对基于静音的分离算法不友好
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p audio in

# 1) 三个说话人（第三个用音高变换从 Tingting 派生）
say -v Tingting -r 190 -o audio/s1.aiff \
  "各位同事，我们今天的周会主要讨论三件事。第一是第四季度的交付节奏，第二是数据平台的迁移方案，第三是下个月的招聘计划。"

say -v Meijia -r 180 -o audio/s2.aiff \
  "我先说一下交付的问题。目前我们有两个模块的进度落后了大概一周，主要是接口联调卡住了，我觉得需要加一个人手。"

say -v Tingting -r 205 -o audio/s3.aiff \
  "我补充一下，数据平台这边我们已经用 ClickHouse 做了压测，查询性能提升了大概三倍。但是我担心迁移期间的双写会带来一致性问题。"

ffmpeg -y -loglevel error -i audio/s3.aiff \
  -af "asetrate=44100*1.12,aresample=44100,atempo=1/1.12" audio/s3_shift.aiff

say -v Meijia -r 185 -o audio/s4.aiff \
  "那这样，交付这块你出一个风险清单，周五之前给我。数据迁移就先做灰度，招聘的事情我下周跟人事对齐。"

# 2) 统一为 16k mono
for f in s1 s2 s3_shift s4; do
  ffmpeg -y -loglevel error -i "audio/$f.aiff" -ar 16000 -ac 1 "audio/$f.wav"
done

# 3) 拼接（无间隔，见文件头说明）
cat > audio/concat.txt <<EOF
file '$PWD/audio/s1.wav'
file '$PWD/audio/s2.wav'
file '$PWD/audio/s3_shift.wav'
file '$PWD/audio/s4.wav'
EOF
ffmpeg -y -loglevel error -f concat -safe 0 -i audio/concat.txt \
  -ar 16000 -ac 1 -c:a pcm_s16le in/meeting_sample.wav

ffprobe -v error -show_entries format=duration,size -of default=nw=1 in/meeting_sample.wav
echo "OK -> bench/in/meeting_sample.wav"
