"""FastAPI：任务列表 / 详情（三轨） / 热词 / 声纹库 / SSE。"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, StreamingResponse)
from fastapi.staticfiles import StaticFiles

from .db import DB
from .engines import FFmpeg, SpeakerEngine
from .openaiapi import build_router
from .settings import Config
from . import __version__ as APP_VERSION
from .worker import ingest

SHELL = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>听记 · TingJi</title><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%23FFE45E' stroke='%231F1A17' stroke-width='3'/%3E%3Crect x='9' y='9' width='14' height='14' rx='4' fill='%23FF4FA3' stroke='%231F1A17' stroke-width='2'/%3E%3C/svg%3E">
<style>
/* 本地打包字体（离线可用，约 80KB，只含拉丁字形；中文回落到苹方） */
@font-face{font-family:"Space Grotesk";src:url("/static/fonts/space-grotesk-700.woff2") format("woff2");
  font-weight:700;font-style:normal;font-display:swap}
@font-face{font-family:"Space Grotesk";src:url("/static/fonts/space-grotesk-500.woff2") format("woff2");
  font-weight:500;font-style:normal;font-display:swap}
@font-face{font-family:"Inter";src:url("/static/fonts/inter-400.woff2") format("woff2");
  font-weight:400;font-style:normal;font-display:swap}
@font-face{font-family:"Inter";src:url("/static/fonts/inter-600.woff2") format("woff2");
  font-weight:600;font-style:normal;font-display:swap}
/* Memphis 设计系统（DESIGN.md v1.0）：奶油底、3px 黑描边、硬阴影、三色撞色 */
:root{
  --bg:#FFF8E8; --surface:#FFFFFF; --ink:#1F1A17; --ink2:#7A5C50;
  --border:#1F1A17; --pink:#FF4FA3; --cyan:#00C2FF; --yellow:#FFE45E;
  --ok:#0E9F6E; --warn:#D98E00; --err:#E03131;
  --r-sm:12px; --r-md:20px; --r-lg:28px;
  --shadow:8px 8px 0 var(--cyan); --shadow-sm:4px 4px 0 var(--cyan);
  /* 兼容旧变量名（内联样式里用到） */
  --card:#FFFFFF; --fg:#1F1A17; --mut:#7A5C50; --acc:#FF4FA3;
}
*{box-sizing:border-box}
body{margin:0;color:var(--ink);font:16px/1.6 Inter,-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
  background:radial-gradient(#F1DFBC 1.3px,transparent 1.4px) 0 0/18px 18px,var(--bg);}
header{position:sticky;top:0;z-index:20;background:var(--yellow);border-bottom:3px solid var(--border);
  padding:12px 24px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
header b{font-family:"Space Grotesk","Avenir Next",Inter,sans-serif;font-size:22px;font-weight:700;letter-spacing:-.02em}
header b:before{content:"";display:inline-block;width:14px;height:14px;margin-right:8px;border-radius:4px;
  background:var(--pink);border:2px solid var(--border);vertical-align:-1px}
header .ver{background:var(--surface);border:2px solid var(--border);border-radius:999px;
  padding:1px 10px;font-size:12px;font-weight:600;color:var(--ink2);font-variant-numeric:tabular-nums}
header a{color:var(--ink);text-decoration:none;font-weight:600;font-size:14px;padding:4px 12px;
  border:2px solid transparent;border-radius:999px}
header a:hover{border-color:var(--border);background:var(--surface)}
header a.on{background:var(--surface);border-color:var(--border);box-shadow:3px 3px 0 var(--pink)}
main{max-width:1100px;margin:0 auto;padding:28px 24px 72px}
h1,h2,h3{font-family:"Space Grotesk","Avenir Next",Inter,sans-serif;letter-spacing:-.02em;margin:0 0 10px}
h1{font-size:40px;line-height:1.08;letter-spacing:-.03em}
h2{font-size:24px;line-height:1.25}h3{font-size:20px}
.card{background:var(--surface);border:3px solid var(--border);border-radius:var(--r-md);
  box-shadow:var(--shadow);padding:24px;margin-bottom:28px}
.card h3{display:flex;align-items:center;gap:8px}
table{width:100%;border-collapse:separate;border-spacing:0}
th{font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:var(--ink2);
  text-align:left;padding:10px 12px;border-bottom:3px solid var(--border)}
td{padding:12px;border-bottom:2px solid #F2E3C6;vertical-align:top}
tr:last-child td{border-bottom:0}
.badge{display:inline-block;padding:1px 10px;border-radius:999px;border:2px solid var(--border);
  background:var(--yellow);font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.04em}
.b-queued{background:#FFF1CF;color:var(--ink2)}
.b-running{background:var(--cyan)}
.b-done{background:#B7F2D2}
.b-failed{background:var(--pink);color:#fff}
.b-clean,.b-final,.b-para,.b-raw{background:var(--surface);border-color:var(--ink2);color:var(--ink2)}
.bar{height:12px;background:var(--surface);border:2px solid var(--border);border-radius:999px;overflow:hidden;width:140px}
.bar>i{display:block;height:100%;background:var(--pink)}
.seg{display:flex;gap:12px;padding:8px;border:2px solid transparent;border-radius:var(--r-sm);cursor:pointer}
.seg:hover{background:#FFF3D2;border-color:var(--border)}
.t{color:var(--ink2);font-variant-numeric:tabular-nums;flex:0 0 128px;font-size:13px}
.spk{color:var(--pink);font-weight:700}
.ov{color:var(--err);font-size:12px}
.w:hover{background:var(--yellow);border-radius:4px}
input,textarea,select{font:inherit;font-size:15px;background:var(--surface);color:var(--ink);
  border:3px solid var(--border);border-radius:var(--r-sm);padding:9px 12px}
input:focus,textarea:focus,select:focus{outline:none;box-shadow:4px 4px 0 var(--cyan)}
button{font:inherit;font-weight:700;cursor:pointer;background:var(--yellow);color:var(--ink);
  border:3px solid var(--border);border-radius:999px;padding:10px 20px;
  box-shadow:4px 4px 0 var(--pink);transition:transform .06s ease,box-shadow .06s ease}
button:hover{transform:translate(-1px,-1px);box-shadow:5px 5px 0 var(--pink)}
button:active{transform:translate(2px,2px);box-shadow:1px 1px 0 var(--pink)}
button.ghost{background:var(--surface);box-shadow:4px 4px 0 var(--cyan)}
button.mini{padding:3px 12px;font-size:12px;border-width:2px;margin-left:6px;box-shadow:2px 2px 0 var(--cyan)}
button.mini.danger{background:var(--surface);color:var(--err);border-color:var(--err);box-shadow:2px 2px 0 var(--pink)}
td.nowrap{white-space:nowrap}
.tabs{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
.tabs button{background:var(--surface);border-width:2px;padding:6px 16px;font-size:13px;box-shadow:none}
.tabs button.on{background:var(--pink);color:#fff;border-color:var(--border);box-shadow:3px 3px 0 var(--border)}
a{color:var(--ink);text-decoration:underline;text-decoration-color:var(--pink);
  text-decoration-thickness:2px;text-underline-offset:3px;font-weight:600}
a:hover{background:var(--yellow)}
input[type=checkbox]{width:18px;height:18px;accent-color:var(--pink);vertical-align:-3px}
input[type=file]{flex:1 1 240px;min-width:240px;max-width:340px;font-size:13px}
::file-selector-button{font:inherit;font-weight:700;cursor:pointer;margin-right:10px;
  background:var(--surface);color:var(--ink);border:2px solid var(--border);
  border-radius:999px;padding:6px 14px}
::file-selector-button:hover{background:var(--yellow)}
td:last-child{padding-right:4px}
.mut{color:var(--ink2)}audio{width:100%;margin:12px 0}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.fields{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:20px}
.field{display:flex;flex-direction:column;gap:6px;min-width:0}
.field>label{font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:var(--ink2)}
.field.wide{grid-column:1/-1}
.hint{font-size:13px;color:var(--ink2);line-height:1.5}
.help{font-size:12px;color:var(--ink2);opacity:.9}
details>summary{cursor:pointer;font-weight:700;font-size:18px;padding:4px 0}
details[open]>summary{margin-bottom:14px}
.flow{display:flex;gap:0;overflow-x:auto;padding:8px 0 4px}
.step{flex:1 0 88px;min-width:88px;text-align:center;position:relative;padding:0 4px}
.step:not(:last-child):after{content:"";position:absolute;top:8px;left:calc(50% + 12px);right:calc(-50% + 12px);
  height:3px;background:var(--border)}
.step.done:not(:last-child):after,.step.skipped:not(:last-child):after{background:var(--pink)}
.step .dot{width:16px;height:16px;border-radius:50%;margin:0 auto 8px;background:var(--surface);
  border:3px solid var(--border);display:block}
.step.done .dot{background:var(--pink)}
.step.skipped .dot{background:#E8D6B2}
.step.running .dot{background:var(--cyan);box-shadow:0 0 0 4px rgba(0,194,255,.28)}
.step.failed .dot{background:var(--err)}
.step .lbl{font-size:12px;font-weight:600;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.step .det{font-size:11px;color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.toast{position:fixed;right:20px;bottom:20px;padding:12px 18px;border-radius:16px;z-index:50;
  background:var(--surface);border:3px solid var(--border);box-shadow:6px 6px 0 var(--cyan);font-weight:600}
.toast.err{border-color:var(--err);box-shadow:6px 6px 0 var(--err)}
.toast.ok{border-color:var(--ok);box-shadow:6px 6px 0 var(--ok)}
.ok{color:var(--ok)}.err{color:var(--err)}
#file{border-style:dashed;background:#FFFDF6}
@media(max-width:720px){
  header{padding:10px 14px;gap:8px}main{padding:16px 14px 56px}
  h1{font-size:30px}.card{padding:16px;border-radius:16px;box-shadow:5px 5px 0 var(--cyan)}
  .t{flex-basis:76px}.fields{grid-template-columns:1fr}table{font-size:13px}th,td{padding:8px}
}
</style></head><body>
<header><b>听记</b><span class="ver">v__VER__</span><a href="/">任务</a><a href="/hotwords">热词</a><a href="/voices">声纹库</a><a href="/logs">日志</a><a href="/settings">设置</a></header>
<main id="app">__BODY__</main>
<script>__JS__</script></body></html>"""

COMMON_JS = """
const $=s=>document.querySelector(s);
const fmt=s=>{s=Math.max(0,Math.floor(s));return [Math.floor(s/3600),Math.floor(s%3600/60),s%60].map(x=>String(x).padStart(2,'0')).join(':')};
const ms=x=>fmt(x/1000);
const esc=s=>(s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
// 统一处理 401：多半是令牌失效/没带 cookie，直接去登录页，避免页面莫名空白
// 进程比代码旧 → 提示重启（客户端请求 401/空白常见根因）
(async()=>{
  try{
    const v=await (await window.fetch('/api/version')).json();
    if(v && v.stale){
      const d=document.createElement('div');
      d.style.cssText='position:sticky;top:0;z-index:99;padding:8px 14px;background:#3a2b12;color:#ffd479;border-bottom:1px solid #6b4e18;font-size:13px';
      d.textContent='代码已更新，但当前服务还是旧进程 —— 请重启服务后刷新页面（否则部分功能不生效）';
      document.body.prepend(d);
    }
  }catch(e){}
})();
const _fetch=window.fetch;
window.fetch=async (...args)=>{
  const r=await _fetch(...args);
  if(r.status===401 && !location.pathname.startsWith('/login')){ location.href='/login?next='+encodeURIComponent(location.pathname); }
  return r;
};
"""


APP_START = __import__("time").time()


def _code_mtime() -> float:
    from .settings import CODE_ROOT
    try:
        return max(p.stat().st_mtime for p in (CODE_ROOT / "app").glob("*.py"))
    except Exception:
        return 0.0


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="TingJi", version="0.1.0")
    db = DB(cfg.db_path)

    def page(body: str, js: str = "") -> HTMLResponse:
        html = SHELL.replace("__BODY__", body).replace("__JS__", COMMON_JS + js)
        return HTMLResponse(html.replace("__VER__", APP_VERSION))

    # ---------------- 任务列表
    # 静态资源（字体）：不需要令牌，登录页也要用
    _static = Path(__file__).resolve().parent / "static"
    if _static.exists():
        app.mount("/static", StaticFiles(directory=str(_static)), name="static")

    # ---------------- 限流（仅对外 OpenAI 兼容接口） ----------------
    from .ratelimit import RateLimiter
    _limiter = RateLimiter(int(getattr(cfg.openai_api, "rate_limit_per_min", 0) or 0))

    @app.middleware("http")
    async def rate_limit_mw(request: Request, call_next):
        if request.url.path.startswith("/v1/"):
            ip = request.client.host if request.client else "unknown"
            if not _limiter.allow(ip):
                return JSONResponse({"detail": "rate limited"}, status_code=429)
        return await call_next(request)

    # ---------------- 访问鉴权（绑定非回环时启用） ----------------
    LOOPBACK = ("127.0.0.1", "localhost", "::1")

    @app.middleware("http")
    async def access_token_mw(request: Request, call_next):
        token = getattr(cfg.server, "token", "") or ""
        if not token or cfg.server.bind in LOOPBACK:
            return await call_next(request)
        path = request.url.path
        # /v1/* 是 OpenAI 兼容接口，用它自己的 api_key 鉴权，不走网页令牌
        # （否则客户端会收到 302 跳登录页，完全没法用）
        if path.startswith("/static/") or path == "/favicon.ico":
            return await call_next(request)     # 字体/图标不鉴权，登录页也要加载
        if path.startswith("/v1/"):
            return await call_next(request)
        if path == "/login" or path == "/api/health":
            return await call_next(request)
        cand = request.query_params.get("token") or request.cookies.get("tingji_token") or ""
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            cand = auth[7:]
        if cand and cand == token:
            resp = await call_next(request)
            # 关键：用 ?token= 打开页面时种下 cookie，否则页面内 fetch 全是 401
            if not request.cookies.get("tingji_token"):
                resp.set_cookie("tingji_token", token, httponly=True,
                                samesite="lax", max_age=30 * 86400)
            return resp
        if path.startswith("/api/"):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return RedirectResponse(f"/login?next={quote(path)}", status_code=302)

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request, token: str = "", next: str = "/", err: str = ""):
        want = getattr(cfg.server, "token", "") or ""
        # next 只允许站内路径；浏览器/反向代理可能把 "/" 编码成 %2F，甚至被改写成 "+"
        from urllib.parse import unquote
        nxt = unquote(next or "/")
        if not nxt.startswith("/") or nxt.startswith("//"):
            nxt = "/"
        if not token and request.cookies.get("tingji_token") == want:
            return RedirectResponse(nxt, status_code=303)   # 已登录，不必再看表单
        if token:
            if token.strip() == want:
                # 303：跳转后用 GET 打开目标页，避免表单重提交；token 不再留在地址栏
                resp = RedirectResponse(nxt, status_code=303)
                resp.set_cookie("tingji_token", want, httponly=True, samesite="lax",
                                max_age=30 * 86400, path="/")
                return resp
            err = "令牌不正确，请重试"
        body = """<div class="card" style="max-width:420px;margin:60px auto">
  <h3>听记 · 需要访问令牌</h3>
  <p class="mut">令牌在服务端启动时打印，或见 <code>config/access_token.txt</code></p>
  """ + (f'<p style="color:#c0392b">{err}</p>' if err else "") + """
  <form method="get" action="/login">
    <input name="token" placeholder="粘贴访问令牌" style="width:100%;padding:8px;margin:8px 0" autofocus>
    <input type="hidden" name="next" value="""" + next + """">
    <button type="submit" style="width:100%">进入</button>
  </form></div>"""
        return page(body, "")

    @app.get("/", response_class=HTMLResponse)
    def index():
        body = """
<div class="card row" style="justify-content:space-between">
  <div><b>任务</b> <span class="mut" id="cnt"></span></div>
  <div class="row"><label class="mut" style="font-size:13px"><input type="checkbox" id="showArch" onchange="load()"> 显示已归档</label>
  <input id="file" type="file" accept="audio/*">
  <button onclick="upload()">上传并处理</button></div>
</div>
<div class="card"><table><thead><tr><th>文件</th><th>时长</th><th>状态</th><th>进度</th><th>阶段</th><th></th></tr></thead>
<tbody id="rows"><tr><td colspan="6" class="mut">加载中…</td></tr></tbody></table></div>"""
        js = """
async function load(){
  const showA=$('#showArch')&&$('#showArch').checked?'1':'0';
  const d=await (await fetch('/api/jobs?archived='+showA)).json();
  $('#cnt').textContent=d.length+' 个任务';
  $('#rows').innerHTML=d.map(j=>`<tr>
    <td>${esc(j.source_name)}</td><td>${(j.duration_sec/60).toFixed(1)}m</td>
    <td><span class="badge b-${j.status}">${j.status}</span></td>
    <td><div class="bar"><i style="width:${j.progress}%"></i></div></td>
    <td class="mut">${j.stage||'-'}<br><small>${esc(j.message||'')}</small></td>
    <td class="nowrap"><a href="/job/${j.id}">查看</a>
      ${j.archived?`<button class="mini" onclick="archiveJob('${j.id}',false)">恢复</button>`
                 :`<button class="mini" onclick="archiveJob('${j.id}',true)">归档</button>`}
      <button class="mini danger" onclick="deleteJob('${j.id}')">删除</button></td></tr>`).join('')
      ||'<tr><td colspan=6 class=mut>暂无任务</td></tr>';
}
async function archiveJob(id, archived){
  await fetch('/api/jobs/'+id+'/archive',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({archived})});
  load();
}
async function deleteJob(id){
  if(!confirm('删除这个任务及其输出？（Obsidian 里已归档的笔记不会被删除）')) return;
  const r=await fetch('/api/jobs/'+id,{method:'DELETE'});
  if(!r.ok) return alert('删除失败');
  load();
}
async function upload(){
  const f=$('#file').files[0]; if(!f) return;
  const fd=new FormData(); fd.append('file',f);
  const r=await fetch('/api/jobs',{method:'POST',body:fd}); const d=await r.json();
  if(!d.queued) alert('未入队：可能是重复文件，或格式不支持');
  load();
}
load(); setInterval(load,3000);
"""
        return page(body, js)

    # ---------------- 任务详情
    @app.get("/job/{jid}", response_class=HTMLResponse)
    def job_page(jid: str):
        body = """
<div class="card" id="head">加载中…</div>
<div class="card" id="audioCard" style="display:none"><audio id="player" controls preload="metadata"></audio></div>
<div class="card" id="flowCard" style="display:none">
  <div class="row" style="justify-content:space-between;margin-bottom:10px">
    <b>处理进度</b><span class="mut" id="flowMeta"></span>
  </div>
  <div class="flow" id="flow"></div>
</div>
<div class="card" id="summaryCard" style="display:none"></div>
<div class="card" id="labelCard" style="display:none">
  <div class="row" style="justify-content:space-between">
    <b>说话人校对</b>
    <span class="mut" id="labelProg"></span>
  </div>
  <p class="mut" style="margin:8px 0">听 5 秒，判断这段话是谁说的。标完会自动算准确率。</p>
  <div id="labelCur" style="font-size:15px;margin:10px 0"></div>
  <div class="row" id="labelBtns"></div>
  <div id="labelResult" class="mut" style="margin-top:12px"></div>
</div>
<div class="card">
  <div class="tabs">
    <button id="tab-para" class="on" onclick="setTrack('para')">整理稿</button>
    <button id="tab-final" onclick="setTrack('final')">逐字稿</button>
    <button id="tab-clean" onclick="setTrack('clean')">清洗稿</button>
    <button id="tab-raw" onclick="setTrack('raw')">原文</button>
    <span class="mut" style="margin-left:12px">点段落可跳转播放</span>
  </div>
  <div id="body"></div>
<div id="splitBox" class="card" style="display:none;position:fixed;left:50%;top:18%;transform:translateX(-50%);
  z-index:60;width:min(560px,92vw);box-shadow:10px 10px 0 var(--cyan)"></div>
</div>"""
        js = """
const JID=location.pathname.split('/').pop();
let DATA=null, TRACK=(new URLSearchParams(location.search).get('track')||'para');
function setTrack(t){
  TRACK=t; ['para','final','clean','raw'].forEach(k=>$('#tab-'+k).classList.toggle('on',k===t));
  try{ const u=new URL(location.href); u.searchParams.set('track',t); history.replaceState(null,'',u); }catch(e){}
  render();
}
function render(){
  if(!DATA) return;
  const rows = TRACK==='para' ? (DATA.paragraphs||[]) : (DATA.segments||[]);
  const key = TRACK==='para' ? 'text' : TRACK;
  $('#body').innerHTML=rows.map((s,i)=>`<div class="seg" data-i="${i}">
    <span class="t">${ms(s.start_ms)}–${ms(s.end_ms)}</span>
    <span class="txt"><span class="spk">${esc(s.speaker)}</span> ${(TRACK==='raw'&&s.words&&s.words.length)
      ? s.words.map(w=>`<span class="w" data-t="${w.s}" title="${ms(w.s)}–${ms(w.e)}${w.approx?'（估算）':''}">${esc(w.t)}</span>`).join('')
      : esc(s[key]||s.text||'')}
    ${s.overlap?'<span class="ov">⚠️重叠</span>':''}
    ${TRACK!=='para'?`<span class="mut" style="margin-left:8px;font-size:12px">
      <a href="#" onclick="event.stopPropagation();setSpk(${s.idx},${s.spk})">改说话人</a> ·
      <a href="#" onclick="event.stopPropagation();mergeNext(${s.idx})">合并下一段</a> ·
      <a href="#" onclick="event.stopPropagation();openSplit(${s.idx})">拆分</a> ·
      <a href="#" onclick="event.stopPropagation();delSeg(${s.idx})">删除</a></span>`:''}
    </span></div>`).join('')||'<span class="mut">还没有结果</span>';
  document.querySelectorAll('.seg').forEach(el=>{
    el.onclick=()=>{const s=rows[+el.dataset.i];const p=$('#player');
      p.currentTime=s.start_ms/1000; p.play();};
  });
  document.querySelectorAll('.seg .w').forEach(el=>{
    el.onclick=(ev)=>{ev.stopPropagation();const p=$('#player');
      p.currentTime=(+el.dataset.t)/1000; p.play();};
  });
  const sm=DATA.summary||{};
  if(sm.summary){
    $('#summaryCard').style.display='block';
    const ch=(sm.chapters||[]).map(c=>`<tr><td class="mut">${ms(c.start_ms||0)}</td><td>${esc(c.title)}</td><td class="mut">${esc(c.summary)}</td></tr>`).join('');
    const td=(sm.todos||[]).map(t=>`<li>${esc(t.task)} — @${esc(t.owner)}</li>`).join('');
    const dc=(sm.decisions||[]).map(d=>`<li>${esc(d)}</li>`).join('');
    const dt=sm.doc_type?`<span class="badge" style="margin-left:8px">${esc(sm.doc_type)}</span>`:'';
    $('#summaryCard').innerHTML=`<div><b>${esc(sm.title||sm.topic||'内容摘要')}</b>${dt}</div>
      <p>${esc(sm.summary)}</p>
      ${ch?`<table><thead><tr><th>时间</th><th>章节</th><th>概要</th></tr></thead><tbody>${ch}</tbody></table>`:''}
      ${dc?`<div style="margin-top:10px"><b>决议</b><ul>${dc}</ul></div>`:''}
      ${td?`<div><b>待办</b><ul>${td}</ul></div>`:''}`;
  }
}
async function loadFlow(){
  try{
    const d=await (await fetch('/api/jobs/'+JID+'/stages')).json();
    if(!d.nodes) return;
    $('#flowCard').style.display='block';
    $('#flowMeta').textContent=`${d.summary.done}/${d.summary.total} · ${d.summary.percent}%`;
    $('#flow').innerHTML=d.nodes.map(n=>`<div class="step ${n.status}" title="${esc(n.desc)}">
      <span class="dot"></span>
      <div class="lbl">${esc(n.label)}</div>
      <div class="det">${n.seconds?n.seconds+'s':''} ${esc(n.detail||'').slice(0,14)}</div>
    </div>`).join('');
  }catch(e){}
}
async function load(){
  loadFlow();
  const r=await fetch('/api/jobs/'+JID); if(!r.ok){$('#head').textContent='任务不存在';return}
  DATA=await r.json(); const j=DATA.job;
  $('#head').innerHTML=`<div class="row" style="justify-content:space-between">
    <div><b>${esc(j.source_name)}</b> <span class="badge b-${j.status}">${j.status}</span>
      <span class="mut">　${(j.duration_sec/60).toFixed(1)} 分钟 · ${j.stage||''} · ${esc(j.message||'')}</span>
      ${DATA.export&&DATA.export.note?`<br><small class="mut">Obsidian: ${esc(DATA.export.note)}</small>`:''}</div>
    <div class="row">
      <select id="fmt" style="width:auto"><option value="srt">SRT</option><option value="vtt">VTT</option><option value="txt">TXT</option></select>
      <button onclick="doExport()">导出</button>
      <button onclick="enrollFromJob()">声纹入库</button>
      <button onclick="rematch()">重新认名</button>
      <button onclick="startLabel()">校对说话人</button>
      <button onclick="rerun()">重跑</button>
    </div></div>
    <div class="bar" style="width:100%;margin-top:10px"><i style="width:${j.progress}%"></i></div>`;
  if(DATA.has_audio){$('#audioCard').style.display='block'; if(!$('#player').src) $('#player').src='/api/audio/'+JID;}
  render();
}
async function rerun(){await fetch('/api/jobs/'+JID+'/rerun',{method:'POST'});load()}
async function enrollFromJob(){
  const spk=prompt('把哪个说话人入库？填编号（0 起）', DATA.segments?0:'0');
  if(spk===null) return;
  const name=prompt('姓名（已存在则追加样本）');
  if(!name) return;
  const r=await fetch('/api/jobs/'+JID+'/enroll',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({spk:parseInt(spk),name})});
  const d=await r.json();
  if(!r.ok){alert(d.detail||'失败');return}
  alert(`已录入「${d.speaker}」${d.added} 段（${d.durations.join('s, ')}s）`);
}
async function rematch(){
  const r=await fetch('/api/jobs/'+JID+'/rematch',{method:'POST'});
  const d=await r.json();
  if(!r.ok){alert(d.detail||'失败');return}
  const n=Object.entries(d.names||{}).map(([k,v])=>`簇${k}→${v}`).join('，')||'没有匹配到任何已登记的人';
  alert('重新认名完成：'+n); load();
}
function doExport(){ const f=$('#fmt').value; window.open('/api/jobs/'+JID+'/export?format='+f,'_blank'); }
async function setSpk(idx, cur){
  const v=prompt('改判为哪个说话人？填编号（0 起）或直接回车保持不变', cur==null?'0':String(cur));
  if(v===null) return;
  const n=parseInt(v); if(isNaN(n)) return alert('请输入数字'); 
  await fetch('/api/jobs/'+JID+'/segments/'+idx,{method:'PATCH',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({speaker:n})});
  alert('已修改，点「重跑」会重建整理稿'); load();
}
function splitCandidates(seg){
  // 从词级时间戳里找"停顿最大"的位置作为候选切点（一句话里有两个人，多半有停顿）
  const ws=(seg.words||[]).filter(w=>w&&w.e!=null);
  if(!ws.length) return [];
  const out=[];
  for(let i=1;i<ws.length;i++){
    const gap=ws[i].s-ws[i-1].e;
    if(gap>=60 && ws[i].s>seg.start_ms+200 && ws[i].e<seg.end_ms-200)
      out.push({t:ws[i].s, gap});
  }
  return out.sort((a,b)=>b.gap-a.gap).slice(0,3);
}
function openSplit(idx){
  const seg=(DATA.segments||[]).find(s=>s.idx===idx); if(!seg) return;
  const cands=splitCandidates(seg);
  const mid=Math.round((seg.start_ms+seg.end_ms)/2);
  const box=$('#splitBox');
  box.innerHTML=`<h3>拆分第 ${idx} 段</h3>
    <div class="hint">${ms(seg.start_ms)}–${ms(seg.end_ms)}　${esc((seg.final_text||seg.raw_text||'').slice(0,60))}</div>
    <div class="row" style="margin:12px 0">${cands.map(c=>
      `<button onclick="doSplit(${idx},${c.t})">在 ${ms(c.t)} 拆（停顿 ${c.gap}ms）</button>`).join('')
      ||'<span class="mut">没有明显的停顿候选，可手动输入时间</span>'}</div>
    <div class="row"><input id="splitAt" type="number" value="${mid}" style="width:160px">
      <span class="mut">毫秒</span><button onclick="doSplit(${idx},Number($('#splitAt').value))">按此拆分</button>
      <button class="ghost" onclick="$('#splitBox').style.display='none'">取消</button></div>
    <div class="hint" style="margin-top:8px">拆分后可分别「改说话人」，再点「重跑」重建整理稿。</div>`;
  box.style.display='block';
}
async function doSplit(idx, at){
  if(!at) return alert('请输入拆分时间（毫秒）');
  const r=await fetch('/api/jobs/'+JID+'/segments/'+idx,{method:'PATCH',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({split_at_ms:at})});
  const d=await r.json();
  if(!r.ok) return alert(d.detail||'拆分失败');
  $('#splitBox').style.display='none';
  load();
}
async function mergeNext(idx){
  await fetch('/api/jobs/'+JID+'/segments/'+idx,{method:'PATCH',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({merge_next:true})}); load();
}
async function delSeg(idx){
  if(!confirm('删除这一段？')) return;
  await fetch('/api/jobs/'+JID+'/segments/'+idx,{method:'PATCH',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({delete:true})}); load();
}
async function delJob(){
  if(!confirm('删除任务及其输出？Obsidian 里已归档的笔记不会动。')) return;
  await fetch('/api/jobs/'+JID,{method:'DELETE'}); location.href='/';
}

// ---------- 说话人校对 ----------
let LQ=[], LP=0, LLABELS=[];
function speakerOptions(){
  const set=new Set((DATA.segments||[]).map(s=>s.speaker).filter(Boolean));
  return [...set].slice(0,6);
}
async function startLabel(){
  const r=await fetch('/api/jobs/'+JID+'/labelqueue?per_cluster=3');
  const d=await r.json(); LQ=d.items||[]; LP=0; LLABELS=[];
  $('#labelCard').style.display='block';
  if(!LQ.length){$('#labelCur').textContent='没有可抽样的片段';return}
  showLabel();
}
function playSeg(seg){
  const p=$('#player'); if(!p.src) p.src='/api/audio/'+JID;
  p.currentTime=seg.start_ms/1000; p.play();
  clearTimeout(window.__stopT);
  window.__stopT=setTimeout(()=>p.pause(), Math.max(1500, (seg.end_ms-seg.start_ms)+400));
}
function showLabel(){
  if(LP>=LQ.length){finishLabel();return}
  const seg=LQ[LP];
  $('#labelProg').textContent=`${LP+1} / ${LQ.length}`;
  $('#labelCur').innerHTML=`<span class="mut">${ms(seg.start_ms)}–${ms(seg.end_ms)}</span>　${esc(seg.text).slice(0,80)}`;
  const opts=speakerOptions();
  $('#labelBtns').innerHTML=opts.map((o,i)=>`<button onclick="pick('${esc(o)}')">${i+1}. ${esc(o)}</button>`).join('')
    + `<button onclick="pick('混合（多人）')">M. 混合（多人）</button>`
    + `<button onclick="pick('导航/噪声')">N. 导航/噪声</button>`
    + `<button onclick="pick('听不清')">S. 听不清</button>`
    + `<button onclick="skipLabel()">跳过</button>`;
  playSeg(seg);
}
function pick(label){ LLABELS.push({idx:LQ[LP].idx,label}); LP++; showLabel(); }
function skipLabel(){ LP++; showLabel(); }
async function finishLabel(){
  $('#labelBtns').innerHTML='';
  $('#labelCur').textContent='标注完成';
  if(!LLABELS.length){$('#labelResult').textContent='没有标注';return}
  const r=await fetch('/api/jobs/'+JID+'/labels',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({items:LLABELS})});
  const a=await r.json();
  const map=Object.entries(a.mapping||{}).map(([k,v])=>`簇${k}→${v}`).join('，');
  const per=Object.entries(a.per_cluster||{}).map(([k,v])=>`簇${k}: ${v.labeled} 段, 纯度 ${v.purity}`).join('；');
  $('#labelResult').innerHTML=`已标注 <b>${a.labeled}</b> 段（其中混合 <b>${a.mixed||0}</b> 段，不计入准确率），自动聚类准确率 <b>${a.accuracy==null?'—':(a.accuracy*100).toFixed(0)+'%'}</b><br>映射：${esc(map)}<br>${esc(per)}`
    + (a.mixed? `<br><span style="color:#c77">混合段已记录，可在下方查看并用于重切</span>`:'');
}
load(); setInterval(load,4000);
"""
        return page(body, js)

    # ---------------- 热词
    @app.get("/logs", response_class=HTMLResponse)
    def logs_page():
        body = """
<div class="card row" style="justify-content:space-between">
  <div><b>运行日志</b> <span class="mut">最近 300 条事件（每 5 秒刷新）</span></div>
  <div class="row">
    <select id="lvl" onchange="load()" style="width:auto">
      <option value="">全部级别</option><option value="info">info</option>
      <option value="warning">warning</option><option value="error">error</option>
    </select>
  </div>
</div>
<div class="card"><table><thead><tr><th style="width:150px">时间</th><th style="width:90px">级别</th>
<th style="width:90px">阶段</th><th style="width:180px">任务</th><th>内容</th></tr></thead>
<tbody id="rows"></tbody></table></div>"""
        js = """
async function load(){
  const lvl=$('#lvl').value;
  const d=await (await fetch('/api/logs?limit=300'+(lvl?'&level='+lvl:''))).json();
  $('#rows').innerHTML=(d.items||[]).map(e=>`<tr>
     <td class="mut">${esc((e.ts||'').replace('T',' ').slice(0,19))}</td>
     <td><span class="badge b-${e.level==='error'?'failed':(e.level==='warning'?'running':'done')}">${esc(e.level||'info')}</span></td>
     <td>${esc(e.stage||'')}</td>
     <td class="mut">${esc(e.source_name||e.job_id||'')}</td>
     <td>${esc(e.message||'')}</td></tr>`).join('')||'<tr><td colspan=5 class=mut>没有日志</td></tr>';
}
load(); setInterval(load, 5000);
"""
        return page(body, js)

    @app.get("/api/logs")
    def api_logs(limit: int = 200, level: str = "", job: str = ""):
        sql = ("SELECT e.*, j.source_name FROM events e LEFT JOIN jobs j ON j.id=e.job_id WHERE 1=1")
        args: list = []
        if level:
            sql += " AND e.level=?"; args.append(level)
        if job:
            sql += " AND e.job_id=?"; args.append(job)
        sql += " ORDER BY e.id DESC LIMIT ?"; args.append(max(1, min(limit, 1000)))
        with db.conn() as c:
            rows = [dict(r) for r in c.execute(sql, args)]
        return {"items": rows}

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page():
        body = """
<div class="card row" style="justify-content:space-between">
  <div>
    <b>设置</b><span class="ver" style="color:#8b93a7;font-size:12px;margin-left:8px">v__VER__</span>
    <div class="hint">改动写入 <code>config/overrides.yaml</code>，立即对新任务生效；<code>server.*</code> 需重启。</div>
  </div>
  <div class="row">
    <button onclick="save()" id="saveBtn">保存</button>
    <button onclick="resetAll()" class="ghost">恢复配置文件默认</button>
  </div>
</div>
<div id="groups"></div>
<details class="card"><summary>环境与模型自检</summary><div id="env" class="hint">检测中…</div></details>
"""
        js = """
let SCHEMA=[], VALUES={}, RESTART=[];
function ctl(item){
  const v=VALUES[item.key];
  const attrs=`data-key="${item.key}"`;
  if(item.type==='bool')
    return `<label class="row" style="gap:8px"><input type="checkbox" ${attrs} ${v?'checked':''}> <span>${item.label}</span></label>`;
  if(item.type==='select'){
    const opts=(item.options||[]).map(o=>`<option value="${esc(o[0])}" ${String(v)===String(o[0])?'selected':''}>${esc(o[1]||o[0])}</option>`).join('');
    return `<select ${attrs}>${opts}</select>`;
  }
  if(item.type==='number')
    return `<input type="number" ${attrs} value="${v}" step="${item.step||1}" min="${item.min??''}" max="${item.max??''}">`;
  return `<input type="text" ${attrs} value="${esc(String(v??''))}" placeholder="${esc(item.placeholder||'')}">`;
}
function field(item){
  const wide=item.wide?'wide':'';
  if(item.type==='bool')
    return `<div class="field ${wide}" data-f="${item.key}">${ctl(item)}
      ${item.help?`<span class="help">${esc(item.help)}</span>`:''}</div>`;
  const test = item.key==='obsidian.vault_path'
    ? ` <button onclick="testVault()" style="align-self:flex-start">测试写入</button>` : '';
  return `<div class="field ${wide}" data-f="${item.key}">
    <label>${esc(item.label)}${item.restart?' <span class="err">(需重启)</span>':''}</label>
    ${ctl(item)}${test}
    ${item.help?`<span class="help">${esc(item.help)}</span>`:''}
    <span class="help err" id="err-${item.key}"></span></div>`;
}
function render(){
  $('#groups').innerHTML=SCHEMA.map((g,i)=>`<div class="card">
    <details ${i<2?'open':''}><summary>${esc(g.group)}</summary>
      <div class="fields">${g.items.map(field).join('')}</div>
    </details></div>`).join('');
}
function collect(){
  const out={};
  document.querySelectorAll('[data-key]').forEach(el=>{
    const k=el.dataset.key;
    if(el.type==='checkbox') out[k]=el.checked;
    else if(el.type==='number') out[k]=el.value===''?null:Number(el.value);
    else out[k]=el.value;
  });
  return out;
}
function toast(msg, ok=true){
  const d=document.createElement('div'); d.className='toast '+(ok?'ok':'err'); d.textContent=msg;
  document.body.appendChild(d); setTimeout(()=>d.remove(), 3200);
}
async function save(){
  document.querySelectorAll('[id^=err-]').forEach(e=>e.textContent='');
  $('#saveBtn').disabled=true;
  try{
    const r=await fetch('/api/settings',{method:'PATCH',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({values:collect()})});
    const d=await r.json();
    if(!d.ok){
      Object.entries(d.errors||{}).forEach(([k,v])=>{ const e=$('#err-'+k); if(e) e.textContent=v; });
      toast('有字段未通过校验', false); return;
    }
    VALUES=d.values; render();
    if(d.warning) toast(d.warning, false);
    else if((d.needs_restart||[]).length) toast('已保存；监听地址/端口需重启服务生效');
    else toast('已保存');
  }finally{ $('#saveBtn').disabled=false; }
}
async function resetAll(){
  if(!confirm('清除 overrides.yaml，恢复 config.yaml 的值？')) return;
  const r=await fetch('/api/settings/reset',{method:'POST'}); const d=await r.json();
  VALUES=d.values; render(); toast('已恢复默认');
}
async function testVault(){
  const path=document.querySelector('[data-key="obsidian.vault_path"]').value;
  const r=await fetch('/api/settings/check-vault',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({path})});
  const d=await r.json();
  toast(d.ok?('可写：'+d.message):('不可用：'+d.message), d.ok);
}
async function load(){
  const d=await (await fetch('/api/settings')).json();
  SCHEMA=d.schema; VALUES=d.values; RESTART=d.restart_keys; render();
  const st=await (await fetch('/api/status')).json();
  const ok=v=>v?'<span class="ok">✓</span>':'<span class="err">✗</span>';
  $('#env').innerHTML=`Python ${esc(st.python)} · ffmpeg ${ok(st.ffmpeg)} · Ollama ${ok(st.ollama)}
    · CPU ${st.cpu} · 内存 ${st.mem_gb} GB
    <br>${Object.entries(st.models||{}).map(([k,v])=>`${esc(k)} ${ok(v.ok)}`).join(' · ')}`;
}
load();
"""
        return page(body, js)

    @app.get("/logs", response_class=HTMLResponse)
    def logs_page():
        body = """
<div class="card row" style="justify-content:space-between">
  <div><b>运行日志</b> <span class="mut">最近 300 条事件（每 5 秒刷新）</span></div>
  <div class="row">
    <select id="lvl" onchange="load()" style="width:auto">
      <option value="">全部级别</option><option value="info">info</option>
      <option value="warning">warning</option><option value="error">error</option>
    </select>
  </div>
</div>
<div class="card"><table><thead><tr><th style="width:150px">时间</th><th style="width:90px">级别</th>
<th style="width:90px">阶段</th><th style="width:180px">任务</th><th>内容</th></tr></thead>
<tbody id="rows"></tbody></table></div>"""
        js = """
async function load(){
  const lvl=$('#lvl').value;
  const d=await (await fetch('/api/logs?limit=300'+(lvl?'&level='+lvl:''))).json();
  $('#rows').innerHTML=(d.items||[]).map(e=>`<tr>
     <td class="mut">${esc((e.ts||'').replace('T',' ').slice(0,19))}</td>
     <td><span class="badge b-${e.level==='error'?'failed':(e.level==='warning'?'running':'done')}">${esc(e.level||'info')}</span></td>
     <td>${esc(e.stage||'')}</td>
     <td class="mut">${esc(e.source_name||e.job_id||'')}</td>
     <td>${esc(e.message||'')}</td></tr>`).join('')||'<tr><td colspan=5 class=mut>没有日志</td></tr>';
}
load(); setInterval(load, 5000);
"""
        return page(body, js)

    @app.get("/api/logs")
    def api_logs(limit: int = 200, level: str = "", job: str = ""):
        sql = ("SELECT e.*, j.source_name FROM events e LEFT JOIN jobs j ON j.id=e.job_id WHERE 1=1")
        args: list = []
        if level:
            sql += " AND e.level=?"; args.append(level)
        if job:
            sql += " AND e.job_id=?"; args.append(job)
        sql += " ORDER BY e.id DESC LIMIT ?"; args.append(max(1, min(limit, 1000)))
        with db.conn() as c:
            rows = [dict(r) for r in c.execute(sql, args)]
        return {"items": rows}

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page():
        body = """
<div class="card"><b>环境与模型</b><div id="env" class="mut" style="margin-top:8px">检测中…</div></div>
<div class="card"><b>关键配置</b><div id="cfg" style="margin-top:8px"></div></div>
<div class="card mut">
配置文件：<code>config/config.yaml</code>（改动后重启服务生效）。模型缺失时对应功能会自动降级。
</div>"""
        js = """
async function load(){
  const d=await (await fetch('/api/status')).json();
  const ok=v=>v?'<span class="badge b-done">就绪</span>':'<span class="badge b-failed">缺失</span>';
  $('#env').innerHTML=`
   <div class="row" style="gap:18px;flex-wrap:wrap">
     <div>Python ${esc(d.python)}</div>
     <div>ffmpeg ${d.ffmpeg?ok(true):ok(false)}</div>
     <div>Ollama ${d.ollama?ok(true):ok(false)} ${esc(d.ollama_error||'')}</div>
     <div>CPU ${d.cpu} 核</div>
     <div>内存 ${d.mem_gb} GB</div>
   </div>
   <table style="margin-top:10px"><thead><tr><th>组件</th><th>状态</th><th>说明</th></tr></thead><tbody>
   ${Object.entries(d.models||{}).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${ok(v.ok)}</td>
     <td class="mut">${esc(v.detail||'')}</td></tr>`).join('')}
   </tbody></table>`;
  $('#cfg').innerHTML=`<table><tbody>
   ${Object.entries(d.config||{}).map(([k,v])=>`<tr><td class="mut">${esc(k)}</td><td>${esc(String(v))}</td></tr>`).join('')}
  </tbody></table>`;
}
load();
"""
        return page(body, js)

    @app.get("/api/settings")
    def api_settings_get():
        from . import settings_ui as sui
        from .settings import OVERRIDES_PATH
        return {
            "schema": sui.SCHEMA,
            "values": sui.values_of(cfg),
            "overrides": OVERRIDES_PATH.exists(),
            "restart_keys": sorted(sui.RESTART_KEYS),
        }

    @app.patch("/api/settings")
    def api_settings_patch(payload: dict):
        from . import settings_ui as sui
        from .settings import save_overrides
        values, errors = sui.validate(payload.get("values") or {})
        if errors:
            return {"ok": False, "errors": errors}
        if not values:
            return {"ok": False, "errors": {"_": "没有可保存的字段"}}
        save_overrides(values)
        sui.reload_into(cfg)
        needs_restart = [k for k in values for g in values[k] if f"{k}.{g}" in sui.RESTART_KEYS]
        # 归档路径变更：立即校验
        warn = ""
        vp = values.get("obsidian", {}).get("vault_path")
        if vp is not None:
            ok, msg = sui.check_vault(vp)
            if not ok:
                warn = f"仓库地址校验未通过：{msg}"
        return {"ok": True, "values": sui.values_of(cfg), "needs_restart": needs_restart,
                "warning": warn}

    @app.post("/api/settings/reset")
    def api_settings_reset():
        from . import settings_ui as sui
        from .settings import clear_overrides
        clear_overrides()
        sui.reload_into(cfg)
        return {"ok": True, "values": sui.values_of(cfg)}

    @app.post("/api/settings/check-vault")
    def api_check_vault(payload: dict):
        from . import settings_ui as sui
        ok, msg = sui.check_vault(str(payload.get("path") or ""))
        return {"ok": ok, "message": msg}

    @app.get("/api/status")
    def api_status():
        import os, shutil, sys, platform
        from .settings import ROOT
        models = {}
        for name, rel in (("FunASR Paraformer", "campplus"), ):
            pass
        paths = {
            "CAM++ 声纹": str(__import__("pathlib").Path(__file__).resolve().parent.parent / ".cache/sherpa/campplus.onnx"),
            "pyannote 分割": str(__import__("pathlib").Path(__file__).resolve().parent.parent / ".cache/sherpa/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx"),
        }
        for label, rel in paths.items():
            p_ = ROOT / rel
            models[label] = {"ok": p_.exists(), "detail": rel if p_.exists() else f"缺少 {rel}"}
        # Ollama
        import urllib.request
        ollama, err = False, ""
        try:
            with urllib.request.urlopen(f"{cfg.llm.base_url.rstrip('/')}/api/tags", timeout=2) as r:
                tags = json.loads(r.read().decode())
            names = [m.get("name") for m in tags.get("models", [])]
            models["Ollama 服务"] = {"ok": True, "detail": f"{len(names)} 个模型"}
            for want in {cfg.correction.model, cfg.summary.model, cfg.roles.model, cfg.refine.model}:
                models[f"模型 {want}"] = {"ok": want in names,
                                          "detail": "已安装" if want in names else "未安装"}
            ollama = True
        except Exception as e:
            models["Ollama 服务"] = {"ok": False, "detail": str(e)[:80]}
            err = str(e)[:80]
        try:
            import psutil
            mem = round(psutil.virtual_memory().total / 1024**3, 1)
        except Exception:
            try:
                mem = round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3, 1)
            except Exception:
                mem = "?"
        return {
            "python": platform.python_version(),
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "ollama": ollama,
            "ollama_error": err,
            "cpu": os.cpu_count(),
            "mem_gb": mem,
            "models": models,
            "config": {
                "asr.provider": getattr(cfg.asr, "provider", "funasr"),
                "asr.model": cfg.asr.model,
                "diarization.backend": getattr(cfg.diarization, "backend", "sherpa"),
                "diarization.match_threshold": getattr(cfg.diarization, "match_threshold", 0.35),
                "diarization.max_speakers": cfg.diarization.max_speakers,
                "text.itn": getattr(cfg.text, "itn", True),
                "refine.enabled": getattr(cfg.refine, "enabled", True),
                "refine.style": getattr(cfg.refine, "style", "auto"),
                "summary.enabled": getattr(cfg.summary, "enabled", True),
                "emotion.enabled": getattr(cfg.emotion, "enabled", True),
                "llm.base_url": cfg.llm.base_url,
                "llm.model(correction)": cfg.correction.model,
                "llm.model(summary)": cfg.summary.model,
                "obsidian.vault_path": str(cfg.obsidian.vault_path) if getattr(cfg.obsidian, "vault_path", None) else "",
                "server.bind": cfg.server.bind,
            },
        }

    @app.get("/hotwords", response_class=HTMLResponse)
    def hotword_page():
        body = """
<div class="card row" style="justify-content:space-between">
  <div><b>热词库</b> <span class="mut">参与 ASR 偏置与文本替换</span></div>
  <div class="row"><input id="term" placeholder="标准写法"><input id="alias" placeholder="别名，逗号分隔" style="width:200px">
  <button onclick="add()">添加</button></div>
</div>
<div class="card"><table><thead><tr><th>热词</th><th>类型</th><th>别名</th><th>权重</th><th></th></tr></thead><tbody id="rows"></tbody></table></div>
<div class="card"><div class="mut">批量导入：每行 <code>词,类型</code> 或 <code>词=>别名1,别名2</code></div>
<textarea id="bulk" rows="5" style="width:100%;margin-top:8px" placeholder="沈知微,人名&#10;河坊街,地名&#10;ClickHouse,英文=>认批,点击house"></textarea>
<div style="margin-top:8px"><button onclick="bulkAdd()">导入</button></div></div>"""
        js = """
async function load(){
  const d=await (await fetch('/api/hotwords')).json();
  $('#rows').innerHTML=d.map(h=>`<tr><td>${esc(h.term)}</td><td class="mut">${esc(h.type)}</td>
    <td class="mut">${(h.aliases||[]).join('、')||'-'}</td><td>${h.weight}</td>
    <td><button onclick="del('${encodeURIComponent(h.term)}')">删除</button></td></tr>`).join('')||'<tr><td colspan=5 class=mut>暂无热词</td></tr>';
}
async function add(){
  const term=$('#term').value.trim(); if(!term) return;
  const aliases=$('#alias').value.split(',').map(s=>s.trim()).filter(Boolean);
  await fetch('/api/hotwords',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({term,aliases})});
  $('#term').value='';$('#alias').value='';load();
}
async function del(t){await fetch('/api/hotwords/'+t,{method:'DELETE'});load()}
async function bulkAdd(){
  for(const l of $('#bulk').value.split('\\n').map(s=>s.trim()).filter(Boolean)){
    let [head, ...rest]=l.split('=>'); let [term,type]=head.split(',').map(s=>s.trim());
    const aliases=rest.join('=>').split(',').map(s=>s.trim()).filter(Boolean);
    if(term) await fetch('/api/hotwords',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({term,type:type||'term',aliases})});
  }
  $('#bulk').value='';load();
}
load();
"""
        return page(body, js)

    # ---------------- 声纹库
    @app.get("/voices", response_class=HTMLResponse)
    def voices_page():
        body = """
<div class="card">
  <div class="row" style="justify-content:space-between">
    <div><b>声纹登记</b> <span class="mut">读 3 句话（约 31 秒），可分段录、分段传</span></div>
    <div class="row"><input id="name" placeholder="姓名" style="min-width:140px">
      <button id="submitBtn" onclick="enroll()">提交注册</button></div>
  </div>
  <p class="hint" style="margin:8px 0">
    用开会同款麦克风、20–40cm、安静环境。每段可现场录音或选择文件；录完可试听、重录。
  </p>
</div>
<div id="slots"></div>
<div class="card" id="result" style="display:none"></div>

<div class="card">
  <b>声纹库管理</b>
  <div class="hint">可以给已有人物追加录音（越多越稳）、改名、删除单条样本或整个人物。</div>
  <table style="margin-top:10px">
    <thead><tr><th>姓名</th><th style="width:70px">样本</th><th style="width:200px">信道</th>
      <th style="width:150px">创建时间</th><th style="width:430px">操作</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
</div>
<input type="file" id="appendFile" accept="audio/*" multiple style="display:none">
<div class="card mut">
提示：至少录 <b>1 段</b>，建议 <b>3 段</b>全录（三句音素覆盖 99.1%）。英文名/生僻字建议同时登记为热词。
认名是保守策略：相似度与领先幅度双阈值，宁可显示"未知"也不给错名字。
</div>"""
        js = """
const SENTENCES=[
 {k:'S1', t:'今天下午三点半，我们在地铁站旁边的小楼里核对预算表，讨论会议内容和交付节奏。', hint:'数字/时间 + 平翘舌 + n-l'},
 {k:'S2', t:'这个季度的目标是把接口响应时间从三百毫秒降到一百二十毫秒，同时保证 API 的稳定性，减少数据波动。', hint:'数字密度 + 英文术语'},
 {k:'S3', t:'如果需要的话，我可以把详细的流程图快速发到群里，大家再决定这一类问题要不要全部采用，顺便跟兄弟部门确认一下情况。', hint:'ü 系列 + uai/ei + r 声母'},
 {k:'FREE', t:'（可选）自由说 15–20 秒：最近在做的一件工作。', hint:'自然语调，纠正朗读腔'}
];
const clips={};
function renderSlots(){
  $('#slots').innerHTML=SENTENCES.map(s=>`
   <div class="card" data-k="${s.k}">
     <div class="row" style="justify-content:space-between">
       <div><b>${s.k}</b> <span class="mut">${esc(s.hint)}</span></div>
       <div class="row">
         <button onclick="recStart('${s.k}')" id="rec-${s.k}">录音</button>
         <button onclick="stopRec('${s.k}')" id="stop-${s.k}" style="display:none">停止</button>
         <input type="file" accept="audio/*" id="file-${s.k}" style="width:150px" onchange="pickFile('${s.k}',this)">
         <button onclick="resetClip('${s.k}')">重录</button>
       </div>
     </div>
     <p style="margin:8px 0">${esc(s.t)}</p>
     <div class="row" id="status-${s.k}"><span class="mut">未录制</span></div>
     <audio id="play-${s.k}" controls style="display:none;width:100%;margin-top:6px"></audio>
   </div>`).join('');
}
function setStatus(k, html){ $('#status-'+k).innerHTML=html; }
function setClip(k, blob, dur){
  clips[k]={blob,dur}; const url=URL.createObjectURL(blob);
  const a=$('#play-'+k); a.src=url; a.style.display='block';
  setStatus(k, `<span class="badge">已录 ${dur.toFixed(1)}s</span>`);
}
let rec=null, recChunks=[], recStartTs=0, recK=null;
async function recStart(k){
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    rec=new MediaRecorder(stream); recK=k; recChunks=[];
    rec.ondataavailable=e=>{ if(e.data.size>0) recChunks.push(e.data); };
    rec.onstop=()=>{ stream.getTracks().forEach(t=>t.stop());
      const blob=new Blob(recChunks,{type:rec.mimeType||'audio/webm'});
      setClip(k, blob, (Date.now()-recStartTs)/1000); };
    rec.start(); recStartTs=Date.now();
    $('#rec-'+k).style.display='none'; $('#stop-'+k).style.display='inline-block';
    setStatus(k,'<span class="badge b-running">录音中…</span>');
  }catch(e){ alert('无法访问麦克风：'+e.message); }
}
function stopRec(k){ if(rec && recK===k) rec.stop();
  $('#rec-'+k).style.display='inline-block'; $('#stop-'+k).style.display='none'; }
function pickFile(k, inp){
  const f=inp.files[0]; if(!f) return;
  const a=$('#play-'+k); const url=URL.createObjectURL(f); a.src=url; a.style.display='block';
  const probe=new Audio(url);
  probe.onloadedmetadata=()=>setClip(k, f, isFinite(probe.duration)?probe.duration:0);
}
function resetClip(k){ delete clips[k]; $('#play-'+k).style.display='none'; setStatus(k,'<span class="mut">未录制</span>'); }
async function enroll(){
  const name=$('#name').value.trim();
  if(!name) return alert('请填写姓名');
  const keys=Object.keys(clips);
  if(!keys.length) return alert('至少录一段');
  const fd=new FormData(); fd.append('name',name);
  for(const k of keys) fd.append('files', clips[k].blob, k+'.webm');
  $('#submitBtn').disabled=true;
  try{
    const r=await fetch('/api/voices',{method:'POST',body:fd}); const d=await r.json();
    if(!d.ok) return alert(d.error||'注册失败');
    showResult(d); $('#name').value='';
    for(const k of keys) resetClip(k);
    load();
  }finally{ $('#submitBtn').disabled=false; }
}
function showResult(d){
  const rows=(d.clips||[]).map(c=>`<tr><td>${c.key}</td><td>${c.ok?'✅':'❌'}</td>
    <td>${c.dur!=null?c.dur.toFixed(1)+'s':'-'}</td><td class="hint">${esc(c.error||'')}</td></tr>`).join('');
  const cons=d.consistency==null?'—':(d.consistency*100).toFixed(0)+'%';
  $('#result').style.display='block';
  $('#result').innerHTML=`<b>注册结果：${esc(d.name)}</b>（有效 ${d.valid}/${d.total}，自一致性 ${cons}）
    <table style="margin-top:8px"><thead><tr><th>段</th><th>状态</th><th>时长</th><th>说明</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}
// ---------------- 管理 ----------------
let SPEAKERS=[];
async function load(){
  SPEAKERS=await (await fetch('/api/voices')).json();
  $('#rows').innerHTML=SPEAKERS.map(v=>`<tr data-sid="${esc(v.id)}">
    <td><b>${esc(v.name)}</b></td><td>${v.samples}</td>
    <td>${(v.channels||1)>1?`<span class="badge b-running">${v.channels} 簇</span>`:`<span class="mut">单信道</span>`}
      ${(v.channel_sizes||[]).map((n,i)=>`<span class="mut" style="font-size:12px">　${i+1}:${n}</span>`).join('')}</td>
    <td class="hint">${esc(v.created_at||'')}</td>
    <td class="row" style="gap:6px">
      <button onclick="append('${esc(v.id)}')">追加录音</button>
      <button onclick="toggleSamples('${esc(v.id)}')">样本</button>
      <button onclick="rename('${esc(v.id)}')">改名</button>
      <button onclick="del('${esc(v.id)}')">删除</button>
    </td></tr>
    <tr id="s-${esc(v.id)}" style="display:none"><td colspan=5><div class="hint">加载中…</div></td></tr>`).join('')
    ||'<tr><td colspan=5 class="mut">还没有注册人物</td></tr>';
}
async function toggleSamples(sid){
  const tr=$('#s-'+sid);
  if(tr.style.display!=='none'){ tr.style.display='none'; return; }
  tr.style.display='table-row';
  const d=await (await fetch('/api/voices/'+sid+'/samples')).json();
  tr.querySelector('td').innerHTML=`<table><thead><tr><th>#</th><th>信道</th><th>质量</th><th>来源</th><th>时间</th><th></th></tr></thead><tbody>
    ${(d.items||[]).map(s=>`<tr><td class="hint">${s.id}</td>
      <td>${s.channel?`<span class="badge b-queued">簇${s.channel}</span>`:'-'}</td>
      <td>${(s.quality??0).toFixed(2)}</td>
      <td class="hint">${esc(s.source||'-')}</td><td class="hint">${esc(s.created_at||'')}</td>
      <td><button onclick="delSample('${sid}',${s.id})">删</button></td></tr>`).join('')}
    </tbody></table>`;
}
async function append(sid){
  const inp=$('#appendFile'); inp.value=''; inp.dataset.sid=sid; inp.click();
}
const _af=$('#appendFile');
if(_af) _af.onchange = async (e)=>{
  const sid=e.target.dataset.sid; const files=e.target.files;
  if(!files.length) return;
  const fd=new FormData();
  for(const f of files) fd.append('files', f, f.name);
  const NL=String.fromCharCode(10);
  const r=await fetch('/api/voices/'+sid+'/samples',{method:'POST',body:fd});
  const d=await r.json();
  if(!r.ok){ alert(d.detail||'追加失败'); return; }
  alert(`已追加 ${d.added} 段，共 ${d.total} 段` + (d.consistency?`；自一致性 ${(d.consistency*100).toFixed(0)}%`:'')
        + (d.warnings&&d.warnings.length?(NL+'⚠️ '+d.warnings.join(NL)):''));
  load();
};
async function rename(sid){
  const oldName=(SPEAKERS.find(x=>x.id===sid)||{}).name||'';
  const name=prompt('新的姓名', oldName);
  if(!name || name===oldName) return;
  const r=await fetch('/api/voices/'+sid,{method:'PATCH',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name})});
  const d=await r.json();
  if(!r.ok){ alert(d.detail||'改名失败'); return; }
  load();
}
async function delSample(sid, vid){
  if(!confirm('删除这条样本？')) return;
  const r=await fetch(`/api/voices/${sid}/samples/${vid}`,{method:'DELETE'});
  const d=await r.json();
  if(!r.ok){ alert(d.detail||'删除失败'); return; }
  const tr=$('#s-'+sid); tr.style.display='none'; toggleSamples(sid); load();
}
async function del(sid){
  if(!confirm('删除该人物及其全部声纹？')) return;
  await fetch('/api/voices/'+sid,{method:'DELETE'}); load();
}
renderSlots(); load();
"""
        return page(body, js)

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page():
        body = """
<div class="card row" style="justify-content:space-between">
  <div>
    <b>设置</b><span class="ver" style="color:#8b93a7;font-size:12px;margin-left:8px">v__VER__</span>
    <div class="hint">改动写入 <code>config/overrides.yaml</code>，立即对新任务生效；<code>server.*</code> 需重启。</div>
  </div>
  <div class="row">
    <button onclick="save()" id="saveBtn">保存</button>
    <button onclick="resetAll()" class="ghost">恢复配置文件默认</button>
  </div>
</div>
<div id="groups"></div>
<details class="card"><summary>环境与模型自检</summary><div id="env" class="hint">检测中…</div></details>
"""
        js = """
let SCHEMA=[], VALUES={}, RESTART=[];
function ctl(item){
  const v=VALUES[item.key];
  const attrs=`data-key="${item.key}"`;
  if(item.type==='bool')
    return `<label class="row" style="gap:8px"><input type="checkbox" ${attrs} ${v?'checked':''}> <span>${item.label}</span></label>`;
  if(item.type==='select'){
    const opts=(item.options||[]).map(o=>`<option value="${esc(o[0])}" ${String(v)===String(o[0])?'selected':''}>${esc(o[1]||o[0])}</option>`).join('');
    return `<select ${attrs}>${opts}</select>`;
  }
  if(item.type==='number')
    return `<input type="number" ${attrs} value="${v}" step="${item.step||1}" min="${item.min??''}" max="${item.max??''}">`;
  return `<input type="text" ${attrs} value="${esc(String(v??''))}" placeholder="${esc(item.placeholder||'')}">`;
}
function field(item){
  const wide=item.wide?'wide':'';
  if(item.type==='bool')
    return `<div class="field ${wide}" data-f="${item.key}">${ctl(item)}
      ${item.help?`<span class="help">${esc(item.help)}</span>`:''}</div>`;
  const test = item.key==='obsidian.vault_path'
    ? ` <button onclick="testVault()" style="align-self:flex-start">测试写入</button>` : '';
  return `<div class="field ${wide}" data-f="${item.key}">
    <label>${esc(item.label)}${item.restart?' <span class="err">(需重启)</span>':''}</label>
    ${ctl(item)}${test}
    ${item.help?`<span class="help">${esc(item.help)}</span>`:''}
    <span class="help err" id="err-${item.key}"></span></div>`;
}
function render(){
  $('#groups').innerHTML=SCHEMA.map((g,i)=>`<div class="card">
    <details ${i<2?'open':''}><summary>${esc(g.group)}</summary>
      <div class="fields">${g.items.map(field).join('')}</div>
    </details></div>`).join('');
}
function collect(){
  const out={};
  document.querySelectorAll('[data-key]').forEach(el=>{
    const k=el.dataset.key;
    if(el.type==='checkbox') out[k]=el.checked;
    else if(el.type==='number') out[k]=el.value===''?null:Number(el.value);
    else out[k]=el.value;
  });
  return out;
}
function toast(msg, ok=true){
  const d=document.createElement('div'); d.className='toast '+(ok?'ok':'err'); d.textContent=msg;
  document.body.appendChild(d); setTimeout(()=>d.remove(), 3200);
}
async function save(){
  document.querySelectorAll('[id^=err-]').forEach(e=>e.textContent='');
  $('#saveBtn').disabled=true;
  try{
    const r=await fetch('/api/settings',{method:'PATCH',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({values:collect()})});
    const d=await r.json();
    if(!d.ok){
      Object.entries(d.errors||{}).forEach(([k,v])=>{ const e=$('#err-'+k); if(e) e.textContent=v; });
      toast('有字段未通过校验', false); return;
    }
    VALUES=d.values; render();
    if(d.warning) toast(d.warning, false);
    else if((d.needs_restart||[]).length) toast('已保存；监听地址/端口需重启服务生效');
    else toast('已保存');
  }finally{ $('#saveBtn').disabled=false; }
}
async function resetAll(){
  if(!confirm('清除 overrides.yaml，恢复 config.yaml 的值？')) return;
  const r=await fetch('/api/settings/reset',{method:'POST'}); const d=await r.json();
  VALUES=d.values; render(); toast('已恢复默认');
}
async function testVault(){
  const path=document.querySelector('[data-key="obsidian.vault_path"]').value;
  const r=await fetch('/api/settings/check-vault',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({path})});
  const d=await r.json();
  toast(d.ok?('可写：'+d.message):('不可用：'+d.message), d.ok);
}
async function load(){
  const d=await (await fetch('/api/settings')).json();
  SCHEMA=d.schema; VALUES=d.values; RESTART=d.restart_keys; render();
  const st=await (await fetch('/api/status')).json();
  const ok=v=>v?'<span class="ok">✓</span>':'<span class="err">✗</span>';
  $('#env').innerHTML=`Python ${esc(st.python)} · ffmpeg ${ok(st.ffmpeg)} · Ollama ${ok(st.ollama)}
    · CPU ${st.cpu} · 内存 ${st.mem_gb} GB
    <br>${Object.entries(st.models||{}).map(([k,v])=>`${esc(k)} ${ok(v.ok)}`).join(' · ')}`;
}
load();
"""
        return page(body, js)

    @app.get("/logs", response_class=HTMLResponse)
    def logs_page():
        body = """
<div class="card row" style="justify-content:space-between">
  <div><b>运行日志</b> <span class="mut">最近 300 条事件（每 5 秒刷新）</span></div>
  <div class="row">
    <select id="lvl" onchange="load()" style="width:auto">
      <option value="">全部级别</option><option value="info">info</option>
      <option value="warning">warning</option><option value="error">error</option>
    </select>
  </div>
</div>
<div class="card"><table><thead><tr><th style="width:150px">时间</th><th style="width:90px">级别</th>
<th style="width:90px">阶段</th><th style="width:180px">任务</th><th>内容</th></tr></thead>
<tbody id="rows"></tbody></table></div>"""
        js = """
async function load(){
  const lvl=$('#lvl').value;
  const d=await (await fetch('/api/logs?limit=300'+(lvl?'&level='+lvl:''))).json();
  $('#rows').innerHTML=(d.items||[]).map(e=>`<tr>
     <td class="mut">${esc((e.ts||'').replace('T',' ').slice(0,19))}</td>
     <td><span class="badge b-${e.level==='error'?'failed':(e.level==='warning'?'running':'done')}">${esc(e.level||'info')}</span></td>
     <td>${esc(e.stage||'')}</td>
     <td class="mut">${esc(e.source_name||e.job_id||'')}</td>
     <td>${esc(e.message||'')}</td></tr>`).join('')||'<tr><td colspan=5 class=mut>没有日志</td></tr>';
}
load(); setInterval(load, 5000);
"""
        return page(body, js)

    @app.get("/api/logs")
    def api_logs(limit: int = 200, level: str = "", job: str = ""):
        sql = ("SELECT e.*, j.source_name FROM events e LEFT JOIN jobs j ON j.id=e.job_id WHERE 1=1")
        args: list = []
        if level:
            sql += " AND e.level=?"; args.append(level)
        if job:
            sql += " AND e.job_id=?"; args.append(job)
        sql += " ORDER BY e.id DESC LIMIT ?"; args.append(max(1, min(limit, 1000)))
        with db.conn() as c:
            rows = [dict(r) for r in c.execute(sql, args)]
        return {"items": rows}

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page():
        body = """
<div class="card"><b>环境与模型</b><div id="env" class="mut" style="margin-top:8px">检测中…</div></div>
<div class="card"><b>关键配置</b><div id="cfg" style="margin-top:8px"></div></div>
<div class="card mut">
配置文件：<code>config/config.yaml</code>（改动后重启服务生效）。模型缺失时对应功能会自动降级。
</div>"""
        js = """
async function load(){
  const d=await (await fetch('/api/status')).json();
  const ok=v=>v?'<span class="badge b-done">就绪</span>':'<span class="badge b-failed">缺失</span>';
  $('#env').innerHTML=`
   <div class="row" style="gap:18px;flex-wrap:wrap">
     <div>Python ${esc(d.python)}</div>
     <div>ffmpeg ${d.ffmpeg?ok(true):ok(false)}</div>
     <div>Ollama ${d.ollama?ok(true):ok(false)} ${esc(d.ollama_error||'')}</div>
     <div>CPU ${d.cpu} 核</div>
     <div>内存 ${d.mem_gb} GB</div>
   </div>
   <table style="margin-top:10px"><thead><tr><th>组件</th><th>状态</th><th>说明</th></tr></thead><tbody>
   ${Object.entries(d.models||{}).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${ok(v.ok)}</td>
     <td class="mut">${esc(v.detail||'')}</td></tr>`).join('')}
   </tbody></table>`;
  $('#cfg').innerHTML=`<table><tbody>
   ${Object.entries(d.config||{}).map(([k,v])=>`<tr><td class="mut">${esc(k)}</td><td>${esc(String(v))}</td></tr>`).join('')}
  </tbody></table>`;
}
load();
"""
        return page(body, js)

    @app.get("/api/settings")
    def api_settings_get():
        from . import settings_ui as sui
        from .settings import OVERRIDES_PATH
        return {
            "schema": sui.SCHEMA,
            "values": sui.values_of(cfg),
            "overrides": OVERRIDES_PATH.exists(),
            "restart_keys": sorted(sui.RESTART_KEYS),
        }

    @app.patch("/api/settings")
    def api_settings_patch(payload: dict):
        from . import settings_ui as sui
        from .settings import save_overrides
        values, errors = sui.validate(payload.get("values") or {})
        if errors:
            return {"ok": False, "errors": errors}
        if not values:
            return {"ok": False, "errors": {"_": "没有可保存的字段"}}
        save_overrides(values)
        sui.reload_into(cfg)
        needs_restart = [k for k in values for g in values[k] if f"{k}.{g}" in sui.RESTART_KEYS]
        # 归档路径变更：立即校验
        warn = ""
        vp = values.get("obsidian", {}).get("vault_path")
        if vp is not None:
            ok, msg = sui.check_vault(vp)
            if not ok:
                warn = f"仓库地址校验未通过：{msg}"
        return {"ok": True, "values": sui.values_of(cfg), "needs_restart": needs_restart,
                "warning": warn}

    @app.post("/api/settings/reset")
    def api_settings_reset():
        from . import settings_ui as sui
        from .settings import clear_overrides
        clear_overrides()
        sui.reload_into(cfg)
        return {"ok": True, "values": sui.values_of(cfg)}

    @app.post("/api/settings/check-vault")
    def api_check_vault(payload: dict):
        from . import settings_ui as sui
        ok, msg = sui.check_vault(str(payload.get("path") or ""))
        return {"ok": ok, "message": msg}

    @app.get("/api/status")
    def api_status():
        import os, shutil, sys, platform
        from .settings import ROOT
        models = {}
        for name, rel in (("FunASR Paraformer", "campplus"), ):
            pass
        paths = {
            "CAM++ 声纹": str(__import__("pathlib").Path(__file__).resolve().parent.parent / ".cache/sherpa/campplus.onnx"),
            "pyannote 分割": str(__import__("pathlib").Path(__file__).resolve().parent.parent / ".cache/sherpa/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx"),
        }
        for label, rel in paths.items():
            p_ = ROOT / rel
            models[label] = {"ok": p_.exists(), "detail": rel if p_.exists() else f"缺少 {rel}"}
        # Ollama
        import urllib.request
        ollama, err = False, ""
        try:
            with urllib.request.urlopen(f"{cfg.llm.base_url.rstrip('/')}/api/tags", timeout=2) as r:
                tags = json.loads(r.read().decode())
            names = [m.get("name") for m in tags.get("models", [])]
            models["Ollama 服务"] = {"ok": True, "detail": f"{len(names)} 个模型"}
            for want in {cfg.correction.model, cfg.summary.model, cfg.roles.model, cfg.refine.model}:
                models[f"模型 {want}"] = {"ok": want in names,
                                          "detail": "已安装" if want in names else "未安装"}
            ollama = True
        except Exception as e:
            models["Ollama 服务"] = {"ok": False, "detail": str(e)[:80]}
            err = str(e)[:80]
        try:
            import psutil
            mem = round(psutil.virtual_memory().total / 1024**3, 1)
        except Exception:
            try:
                mem = round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3, 1)
            except Exception:
                mem = "?"
        return {
            "python": platform.python_version(),
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "ollama": ollama,
            "ollama_error": err,
            "cpu": os.cpu_count(),
            "mem_gb": mem,
            "models": models,
            "config": {
                "asr.provider": getattr(cfg.asr, "provider", "funasr"),
                "asr.model": cfg.asr.model,
                "diarization.backend": getattr(cfg.diarization, "backend", "sherpa"),
                "diarization.match_threshold": getattr(cfg.diarization, "match_threshold", 0.35),
                "diarization.max_speakers": cfg.diarization.max_speakers,
                "text.itn": getattr(cfg.text, "itn", True),
                "refine.enabled": getattr(cfg.refine, "enabled", True),
                "refine.style": getattr(cfg.refine, "style", "auto"),
                "summary.enabled": getattr(cfg.summary, "enabled", True),
                "emotion.enabled": getattr(cfg.emotion, "enabled", True),
                "llm.base_url": cfg.llm.base_url,
                "llm.model(correction)": cfg.correction.model,
                "llm.model(summary)": cfg.summary.model,
                "obsidian.vault_path": str(cfg.obsidian.vault_path) if getattr(cfg.obsidian, "vault_path", None) else "",
                "server.bind": cfg.server.bind,
            },
        }

    @app.get("/hotwords", response_class=HTMLResponse)
    def hotword_page():
        body = """
<div class="card row" style="justify-content:space-between">
  <div><b>热词库</b> <span class="mut">参与 ASR 偏置与文本替换</span></div>
  <div class="row"><input id="term" placeholder="标准写法"><input id="alias" placeholder="别名，逗号分隔" style="width:200px">
  <button onclick="add()">添加</button></div>
</div>
<div class="card"><table><thead><tr><th>热词</th><th>类型</th><th>别名</th><th>权重</th><th></th></tr></thead><tbody id="rows"></tbody></table></div>
<div class="card"><div class="mut">批量导入：每行 <code>词,类型</code> 或 <code>词=>别名1,别名2</code></div>
<textarea id="bulk" rows="5" style="width:100%;margin-top:8px" placeholder="沈知微,人名&#10;河坊街,地名&#10;ClickHouse,英文=>认批,点击house"></textarea>
<div style="margin-top:8px"><button onclick="bulkAdd()">导入</button></div></div>"""
        js = """
async function load(){
  const d=await (await fetch('/api/hotwords')).json();
  $('#rows').innerHTML=d.map(h=>`<tr><td>${esc(h.term)}</td><td class="mut">${esc(h.type)}</td>
    <td class="mut">${(h.aliases||[]).join('、')||'-'}</td><td>${h.weight}</td>
    <td><button onclick="del('${encodeURIComponent(h.term)}')">删除</button></td></tr>`).join('')||'<tr><td colspan=5 class=mut>暂无热词</td></tr>';
}
async function add(){
  const term=$('#term').value.trim(); if(!term) return;
  const aliases=$('#alias').value.split(',').map(s=>s.trim()).filter(Boolean);
  await fetch('/api/hotwords',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({term,aliases})});
  $('#term').value='';$('#alias').value='';load();
}
async function del(t){await fetch('/api/hotwords/'+t,{method:'DELETE'});load()}
async function bulkAdd(){
  for(const l of $('#bulk').value.split('\\n').map(s=>s.trim()).filter(Boolean)){
    let [head, ...rest]=l.split('=>'); let [term,type]=head.split(',').map(s=>s.trim());
    const aliases=rest.join('=>').split(',').map(s=>s.trim()).filter(Boolean);
    if(term) await fetch('/api/hotwords',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({term,type:type||'term',aliases})});
  }
  $('#bulk').value='';load();
}
load();
"""
        return page(body, js)

    # ---------------- 声纹库
    @app.get("/voices", response_class=HTMLResponse)
    def voices_page():
        body = """
<div class="card">
  <div class="row" style="justify-content:space-between">
    <div><b>声纹登记</b> <span class="mut">读 3 句话（约 31 秒），可分段录、分段传</span></div>
    <div class="row"><input id="name" placeholder="姓名" style="min-width:140px">
      <button id="submitBtn" onclick="enroll()">提交注册</button></div>
  </div>
  <p class="mut" style="margin:8px 0">
    用开会同款麦克风、20–40cm、安静环境。每段可现场录音或选择文件；录完可试听、重录。
  </p>
</div>
<div id="slots"></div>
<div class="card" id="result" style="display:none"></div>
<div class="card"><table><thead><tr><th>姓名</th><th>样本数</th><th>创建时间</th><th></th></tr></thead><tbody id="rows"></tbody></table></div>
<div class="card mut">
提示：至少录 <b>1 段</b>，建议 <b>3 段</b>全录（三句音素覆盖 99.1%）。英文名/生僻字建议同时登记为热词。
注册后下次转写会自动认名；出现"宁可显示不知道，也不要显示错人名"的保守策略（相似度与领先幅度双阈值）。
</div>"""
        js = """
const SENTENCES=[
 {k:'S1', t:'今天下午三点半，我们在地铁站旁边的小楼里核对预算表，讨论会议内容和交付节奏。', hint:'数字/时间 + 平翘舌 + n-l'},
 {k:'S2', t:'这个季度的目标是把接口响应时间从三百毫秒降到一百二十毫秒，同时保证 API 的稳定性，减少数据波动。', hint:'数字密度 + 英文术语'},
 {k:'S3', t:'如果需要的话，我可以把详细的流程图快速发到群里，大家再决定这一类问题要不要全部采用，顺便跟兄弟部门确认一下情况。', hint:'ü 系列 + uai/ei + r 声母'},
 {k:'FREE', t:'（可选）自由说 15–20 秒：最近在做的一件工作。', hint:'自然语调，纠正朗读腔'}
];
const clips={};   // k -> {blob, dur, url}

function renderSlots(){
  $('#slots').innerHTML=SENTENCES.map((s,i)=>`
   <div class="card" data-k="${s.k}">
     <div class="row" style="justify-content:space-between">
       <div><b>${s.k}</b> <span class="mut">${esc(s.hint)}</span></div>
       <div class="row">
         <button onclick="recStart('${s.k}')" id="rec-${s.k}">录音</button>
         <button onclick="stopRec('${s.k}')" id="stop-${s.k}" style="display:none">停止</button>
         <input type="file" accept="audio/*" id="file-${s.k}" style="width:150px" onchange="pickFile('${s.k}',this)">
         <button onclick="resetClip('${s.k}')">重录</button>
       </div>
     </div>
     <p style="margin:8px 0">${esc(s.t)}</p>
     <div class="row" id="status-${s.k}"><span class="mut">未录制</span></div>
     <audio id="play-${s.k}" controls style="display:none;width:100%;margin-top:6px"></audio>
   </div>`).join('');
}
function setStatus(k, html){ $('#status-'+k).innerHTML=html; }
function setClip(k, blob, dur){
  clips[k]={blob,dur}; const url=URL.createObjectURL(blob);
  const a=$('#play-'+k); a.src=url; a.style.display='block';
  setStatus(k, `<span class="badge">已录 ${dur.toFixed(1)}s</span>`);
}
let rec=null, recChunks=[], recStartTs=0, recK=null;
async function recStart(k){
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    rec=new MediaRecorder(stream); recK=k; recChunks=[];
    rec.ondataavailable=e=>{ if(e.data.size>0) recChunks.push(e.data); };
    rec.onstop=async ()=>{
      stream.getTracks().forEach(t=>t.stop());
      const blob=new Blob(recChunks,{type:rec.mimeType||'audio/webm'});
      const dur=(Date.now()-recStartTs)/1000;
      setClip(k, blob, dur);
    };
    rec.start(); recStartTs=Date.now();
    $('#rec-'+k).style.display='none'; $('#stop-'+k).style.display='inline-block';
    setStatus(k,'<span class="badge b-running">录音中…</span>');
  }catch(e){ alert('无法访问麦克风：'+e.message); }
}
function stopRec(k){ if(rec && recK===k){ rec.stop(); } $('#rec-'+k).style.display='inline-block'; $('#stop-'+k).style.display='none'; }
function pickFile(k, inp){
  const f=inp.files[0]; if(!f) return;
  const a=$('#play-'+k); const url=URL.createObjectURL(f); a.src=url; a.style.display='block';
  const probe=new Audio(url);
  probe.onloadedmetadata=()=>setClip(k, f, isFinite(probe.duration)?probe.duration:0);
}
function resetClip(k){ delete clips[k]; $('#play-'+k).style.display='none'; setStatus(k,'<span class="mut">未录制</span>'); }
async function enroll(){
  const name=$('#name').value.trim();
  if(!name) return alert('请填写姓名');
  const keys=Object.keys(clips);
  if(!keys.length) return alert('至少录一段');
  const fd=new FormData(); fd.append('name',name);
  for(const k of keys) fd.append('files', clips[k].blob, k+'.webm');
  $('#submitBtn').disabled=true;
  try{
    const r=await fetch('/api/voices',{method:'POST',body:fd}); const d=await r.json();
    if(!d.ok) return alert(d.error||'注册失败');
    showResult(d); $('#name').value='';
    for(const k of keys) resetClip(k);
    load();
  }finally{ $('#submitBtn').disabled=false; }
}
function showResult(d){
  const rows=(d.clips||[]).map(c=>`<tr><td>${c.key}</td><td>${c.ok?'✅':'❌'}</td>
    <td>${c.dur!=null?c.dur.toFixed(1)+'s':'-'}</td><td class="mut">${esc(c.error||'')}</td></tr>`).join('');
  const cons=d.consistency==null?'—':(d.consistency*100).toFixed(0)+'%';
  const warn=(d.warnings||[]).map(w=>`<li>${esc(w)}</li>`).join('');
  $('#result').style.display='block';
  $('#result').innerHTML=`<b>注册结果：${esc(d.name)}</b>（有效样本 ${d.valid}/${d.total}，三句自一致性 ${cons}）
    <table style="margin-top:8px"><thead><tr><th>段</th><th>状态</th><th>时长</th><th>说明</th></tr></thead><tbody>${rows}</tbody></table>
    ${warn?`<ul class="mut" style="margin-top:8px">${warn}</ul>`:''}`;
}
async function load(){
  const d=await (await fetch('/api/voices')).json();
  $('#rows').innerHTML=d.map(v=>`<tr><td>${esc(v.name)}</td><td>${v.samples}</td>
    <td class="mut">${v.created_at||''}</td>
    <td><button onclick="del('${v.id}')">删除</button></td></tr>`).join('')||'<tr><td colspan=4 class=mut>还没有注册人物</td></tr>';
}
async function del(id){ if(!confirm('删除该人物及其声纹？')) return;
  await fetch('/api/voices/'+id,{method:'DELETE'}); load(); }
renderSlots(); load();
"""
        return page(body, js)

    @app.get("/api/jobs")
    def api_jobs(archived: int = 0):
        return db.list_jobs(include_archived=bool(archived))

    @app.post("/api/jobs/{jid}/archive")
    def api_archive_job(jid: str, body: dict | None = None):
        job = db.get_job(jid)
        if not job:
            raise HTTPException(404)
        flag = bool((body or {}).get("archived", True))
        db.set_archived(jid, flag)
        db.log(jid, "info", "archive", "已归档" if flag else "已取消归档")
        return {"ok": True, "archived": flag}

    @app.get("/api/jobs/{jid}")
    def api_job(jid: str):
        job = db.get_job(jid)
        if not job:
            raise HTTPException(404, "job not found")
        segs = db.get_segments(jid)
        outdir = cfg.output_path / Path(job["source_name"]).stem
        exp = db.get_export(jid) or {}
        # 说话人显示名：优先 speakers.json 中的认名
        names: dict[int, str] = {}
        sp = cfg.work_path / jid / "speakers.json"
        if sp.exists():
            try:
                names = {int(k): v for k, v in json.loads(sp.read_text())["names"].items()}
            except Exception:
                names = {}
        per: dict[int, float] = {}
        for s in segs:
            if s["spk"] is not None and s["spk"] >= 0:
                per[s["spk"]] = per.get(s["spk"], 0) + (s["end_ms"] - s["start_ms"]) / 1000
        order = sorted(per, key=lambda k: -per[k])
        renum = {k: i + 1 for i, k in enumerate(order)}
        paras = db.get_paragraphs(jid)
        return {
            "job": job,
            "export": exp,
            "summary": db.get_summary(jid) or {},
            "paragraphs": [{"idx": p["idx"], "start_ms": p["start_ms"], "end_ms": p["end_ms"],
                            "speaker": names.get(p["spk"]) or (f"说话人{renum.get(p['spk'], (p['spk'] or 0) + 1)}" if p["spk"] is not None and p["spk"] >= 0 else "未知"),
                            "text": p["text"], "polished": bool(p["polished"])} for p in paras],
            "segments": [{
                "idx": s["idx"], "start_ms": s["start_ms"], "end_ms": s["end_ms"],
                "speaker": (names.get(s["spk"]) if s["spk"] is not None else None)
                           or (f"说话人{renum.get(s['spk'], (s['spk'] or 0) + 1)}" if s["spk"] is not None and s["spk"] >= 0 else "未知"),
                "final": s["final_text"] or s["clean_text"] or s["raw_text"],
                "clean": s["clean_text"] or s["raw_text"],
                "raw": s["raw_text"],
                "overlap": bool(s["overlap"]),
                "spk": s["spk"],
                "words": json.loads(s["words"]) if s.get("words") else [],
                "boundary": s.get("boundary"),
            } for s in segs],
            "has_audio": (outdir / "preview.m4a").exists(),
        }

    @app.get("/api/jobs/{jid}/labelqueue")
    def api_label_queue(jid: str, per_cluster: int = 3):
        if not db.get_job(jid):
            raise HTTPException(404, "job not found")
        from . import labeling
        return {"items": labeling.label_queue(db.get_segments(jid), per_cluster=per_cluster)}

    @app.post("/api/jobs/{jid}/labels")
    def api_save_labels(jid: str, payload: dict):
        items = payload.get("items") or []
        if not items:
            raise HTTPException(400, "items 为空")
        db.save_seg_labels(jid, items)
        from . import labeling
        return labeling.accuracy(db.get_segments(jid), db.get_seg_labels(jid))

    @app.get("/api/jobs/{jid}/accuracy")
    def api_accuracy(jid: str):
        from . import labeling
        return labeling.accuracy(db.get_segments(jid), db.get_seg_labels(jid))

    @app.get("/api/jobs/{jid}/mixed")
    def api_mixed(jid: str):
        from .labeling import MIXED
        segs = {int(x["idx"]): x for x in db.get_segments(jid)}
        out = []
        for idx, lab in db.get_seg_labels(jid).items():
            if lab == MIXED and int(idx) in segs:
                s_ = segs[int(idx)]
                out.append({"idx": int(idx), "start_ms": s_["start_ms"], "end_ms": s_["end_ms"],
                            "spk": s_.get("spk"), "text": s_.get("raw_text")})
        return {"items": sorted(out, key=lambda x: x["start_ms"])}

    @app.delete("/api/jobs/{jid}/labels")
    def api_clear_labels(jid: str):
        db.clear_seg_labels(jid)
        return {"ok": True}

    @app.post("/api/jobs")
    async def api_upload(file: UploadFile = File(...)):
        dest = cfg.input_path / file.filename
        dest.write_bytes(await file.read())
        jid = ingest(db, cfg, dest)
        return {"id": jid, "queued": bool(jid)}

    @app.post("/api/jobs/{jid}/rerun")
    def api_rerun(jid: str):
        if not db.get_job(jid):
            raise HTTPException(404)
        db.update_job(jid, status="queued", progress=0, stage=None, message="重新排队", error_code=None)
        return {"ok": True}

    @app.patch("/api/jobs/{jid}/segments/{idx}")
    async def api_patch_segment(jid: str, idx: int, req: Request):
        """手动干预（FR-C4）：改文字 / 改说话人 / 与下一段合并 / 删除本段。"""
        body = await req.json()
        segs = db.get_segments(jid)
        cur = next((x for x in segs if int(x["idx"]) == int(idx)), None)
        if cur is None:
            raise HTTPException(404, "segment not found")

        if body.get("delete"):
            db.delete_segment(jid, idx)
            db.log(jid, "info", "manual", f"删除第 {idx} 段")
            return {"ok": True, "rebuild": True}

        if body.get("split_at_ms"):
            from . import textproc as _tp
            at = int(body["split_at_ms"])
            words = []
            try:
                words = json.loads(cur["words"]) if cur.get("words") else []
            except Exception:
                words = []
            text = cur["final_text"] or cur["clean_text"] or cur["raw_text"] or ""
            try:
                left, right = _tp.split_by_time(words, int(cur["start_ms"]), int(cur["end_ms"]), at, text)
            except ValueError as e:
                raise HTTPException(400, str(e))
            if not left.strip() or not right.strip():
                raise HTTPException(400, "拆分后有一侧为空，换个位置再试")
            lw = [w for w in words if int(w.get("e", w.get("s", 0))) <= at] or None
            rw = [w for w in words if int(w.get("e", w.get("s", 0))) > at] or None
            try:
                new_idx = db.split_segment(jid, idx, at, left.strip(), right.strip(), lw, rw)
            except ValueError as e:
                raise HTTPException(400, str(e))
            db.log(jid, "info", "manual", f"拆分第 {idx} 段（{at/1000:.1f}s）→ 新段 {new_idx}")
            return {"ok": True, "rebuild": True, "new_idx": new_idx,
                    "left": left.strip(), "right": right.strip()}

        if body.get("merge_next"):
            if not db.merge_segment_next(jid, idx):
                raise HTTPException(400, "没有下一段")
            db.log(jid, "info", "manual", f"合并第 {idx} 与 {idx + 1} 段")
            return {"ok": True, "rebuild": True}

        if "speaker" in body:
            with db.conn() as c:
                c.execute("UPDATE segments SET spk=? WHERE job_id=? AND idx=?",
                          (int(body["speaker"]), jid, idx))
            db.log(jid, "info", "manual", f"第 {idx} 段改判为说话人 {int(body['speaker']) + 1}")
            return {"ok": True, "rebuild": True}

        if "text" not in body:
            raise HTTPException(400, "text / speaker / merge_next / delete 至少给一个")
        db.update_segment(jid, idx, final_text=body["text"])
        return {"ok": True, "rebuild": False}

    @app.post("/api/jobs/{jid}/enroll")
    def api_enroll_from_job(jid: str, payload: dict):
        """把任务里某个说话人回写进声纹库（FR-C5），免去重新录音。"""
        name = (payload.get("name") or "").strip()
        try:
            spk_id = int(payload.get("spk"))
        except Exception:
            raise HTTPException(400, "缺少 spk")
        if not name:
            raise HTTPException(400, "缺少姓名")
        segs = [x for x in db.get_segments(jid)
                if x["spk"] is not None and int(x["spk"]) == spk_id]
        if not segs:
            raise HTTPException(404, "该说话人没有可用片段")
        # 挑最干净的长段（purity 优先、时长其次），最多 3 段
        def score(x):
            pur = float(x.get("spk_purity") or 0)
            dur = (x["end_ms"] - x["start_ms"]) / 1000
            return (pur >= 0.99, dur)
        picks = sorted(segs, key=score, reverse=True)[:3]
        wav = cfg.work_path / jid / "16k.wav"
        if not wav.exists():
            raise HTTPException(400, "缺少音频")
        import soundfile as sf
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        eng = SpeakerEngine(cfg.diarization.campplus_model)
        vecs, used = [], []
        for x in picks:
            a, b = int(x["start_ms"] / 1000 * sr), int(x["end_ms"] / 1000 * sr)
            clip = audio[a:b]
            if len(clip) < int(1.0 * sr):
                continue
            v = eng.embed(clip)
            if v is not None:
                vecs.append(np.asarray(v, dtype=np.float32))
                used.append(round((x["end_ms"] - x["start_ms"]) / 1000, 1))
        if not vecs:
            raise HTTPException(400, "片段太短，无法提取声纹")
        existing = db.get_speaker_by_name(name)
        sid = existing["id"] if existing else db.create_speaker(name)
        for v in vecs:
            db.add_voiceprint(sid, v.tobytes(), int(v.shape[0]), source="job", quality=1.0)
        db.log(jid, "info", "voices", f"声纹回写：{name} +{len(vecs)} 段")
        return {"ok": True, "speaker": name, "added": len(vecs), "durations": used}

    @app.post("/api/jobs/{jid}/rematch")
    def api_rematch(jid: str):
        """新增声纹后，不重跑识别，直接对已有结果重新认名。"""
        job = db.get_job(jid)
        if not job:
            raise HTTPException(404)
        segs = db.get_segments(jid)
        if not segs:
            raise HTTPException(400, "还没有转写结果")
        wdir = cfg.work_path / jid
        wav = wdir / "16k.wav"
        if not wav.exists():
            raise HTTPException(400, "缺少音频，无法提取声纹")
        import soundfile as sf
        from . import diarize as dia
        from .pipeline import match_voiceprints
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        spk = SpeakerEngine(cfg.diarization.campplus_model)
        embs = []
        for x in segs:
            a, b = int(x["start_ms"] / 1000 * sr), int(x["end_ms"] / 1000 * sr)
            clip = audio[a:b]
            embs.append(spk.embed(clip) if len(clip) > int(0.3 * sr) else None)
        E = np.stack([e if e is not None else np.zeros(192, dtype=np.float32) for e in embs])
        labels = np.array([int(x["spk"]) if x["spk"] is not None else -1 for x in segs])
        cents = dia.centroids_clean(E, labels, segs)
        names, matches = match_voiceprints(
            db, cents,
            threshold=float(getattr(cfg.diarization, "match_threshold", 0.35)),
            margin=float(getattr(cfg.diarization, "match_margin", 0.10)))
        sp = wdir / "speakers.json"
        data = json.loads(sp.read_text()) if sp.exists() else {}
        data["names"] = {str(k): v for k, v in names.items()}
        data["matches"] = matches
        sp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        db.log(jid, "info", "identified", f"重新认名 {len(names)} 人", matches)
        return {"ok": True, "names": {str(k): v for k, v in names.items()},
                "matches": matches}

    @app.post("/api/jobs/{jid}/rebuild")
    def api_rebuild(jid: str):
        """只重跑后半段（整理稿/纪要/归档），不重新识别。"""
        job = db.get_job(jid)
        if not job:
            raise HTTPException(404)
        wdir = cfg.work_path / jid
        st = wdir / "state.json"
        if st.exists():
            try:
                data = json.loads(st.read_text())
                done = [x for x in data.get("done", [])
                        if x not in ("refined", "roles", "summarized", "archived", "corrected")]
                st.write_text(json.dumps({**data, "done": done}, ensure_ascii=False))
            except Exception:
                pass
        for f in ("paragraphs.json",):
            (wdir / f).unlink(missing_ok=True)
        db.update_job(jid, status="queued", progress=0, message="等待重跑整理稿")
        return {"ok": True}

    @app.get("/api/jobs/{jid}/export")
    def api_export(jid: str, format: str = "srt", track: str = "final", level: str = "segment"):
        from fastapi.responses import Response
        job = db.get_job(jid)
        if not job:
            raise HTTPException(404)
        segs = db.get_segments(jid)
        if not segs:
            raise HTTPException(404, "还没有转写结果")
        names: dict[int, str] = {}
        sp = cfg.work_path / jid / "speakers.json"
        if sp.exists():
            try:
                names = {int(k): v for k, v in json.loads(sp.read_text())["names"].items()}
            except Exception:
                names = {}
        per: dict[int, float] = {}
        for x in segs:
            if x["spk"] is not None and x["spk"] >= 0:
                per[x["spk"]] = per.get(x["spk"], 0) + (x["end_ms"] - x["start_ms"]) / 1000
        order = sorted(per, key=lambda k: -per[k])
        renum = {k: i + 1 for i, k in enumerate(order)}

        def label(x):
            k = x["spk"]
            if k is None or k < 0:
                return "未知"
            return names.get(k) or f"说话人{renum.get(k, k + 1)}"

        def text_of(x):
            if track == "raw":
                return x["raw_text"] or ""
            if track == "clean":
                return x["clean_text"] or x["raw_text"] or ""
            return x["final_text"] or x["clean_text"] or x["raw_text"] or ""

        def hms(ms, sep=","):
            h, r = divmod(max(0, int(ms)), 3600000)
            m, r = divmod(r, 60000)
            s, ms2 = divmod(r, 1000)
            return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms2:03d}"

        base = Path(job["source_name"]).stem
        if level == "word" and format in ("srt", "vtt"):
            lines = ["WEBVTT", ""] if format == "vtt" else []
            n = 0
            for x in segs:
                try:
                    ws = json.loads(x["words"]) if x.get("words") else []
                except Exception:
                    ws = []
                for w in ws:
                    n += 1
                    tag = f"[{label(x)}] "
                    if format == "srt":
                        lines += [str(n), f"{hms(w['s'])} --> {hms(w['e'])}", f"{tag}{w['t']}", ""]
                    else:
                        lines += [f"{hms(w['s'], '.')} --> {hms(w['e'], '.')}", f"{tag}{w['t']}", ""]
            body = "\n".join(lines)
            media = "text/vtt; charset=utf-8" if format == "vtt" else "application/x-subrip; charset=utf-8"
            fname = quote(f"{base}-{track}-word.{format}")
            return Response(body, media_type=media,
                            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{fname}"})

        if format in ("srt", "vtt"):
            lines = ["WEBVTT", ""] if format == "vtt" else []
            for i, x in enumerate(segs, 1):
                t = text_of(x).strip()
                if not t:
                    continue
                if format == "srt":
                    lines += [str(i), f"{hms(x['start_ms'])} --> {hms(x['end_ms'])}", f"[{label(x)}] {t}", ""]
                else:
                    lines += [f"{hms(x['start_ms'], '.')} --> {hms(x['end_ms'], '.')}",
                              f"<v {label(x)}>{t}", ""]
            body = "\n".join(lines)
            media = "text/vtt; charset=utf-8" if format == "vtt" else "application/x-subrip; charset=utf-8"
            ext = format
        else:
            out = []
            for x in segs:
                t = text_of(x).strip()
                if t:
                    out.append(f"[{label(x)}] {t}")
            body = "\n".join(out)
            media = "text/plain; charset=utf-8"
            ext = "txt"
        # 文件名可能是中文，必须用 RFC 5987 编码，否则 latin-1 头会 500
        fname = quote(f"{base}-{track}.{ext}")
        return Response(body, media_type=media,
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{fname}"})

    @app.delete("/api/jobs/{jid}")
    def api_delete_job(jid: str, keep_archive: int = 0):
        job = db.get_job(jid)
        if not job:
            raise HTTPException(404)
        import shutil
        if not keep_archive:
            outdir = cfg.output_path / Path(job["source_name"]).stem
            shutil.rmtree(outdir, ignore_errors=True)
        shutil.rmtree(cfg.work_path / jid, ignore_errors=True)
        db.delete_job(jid)
        return {"ok": True, "kept_archive": bool(keep_archive)}

    @app.get("/api/audio/{jid}")
    def api_audio(jid: str):
        job = db.get_job(jid)
        if not job:
            raise HTTPException(404)
        p = cfg.output_path / Path(job["source_name"]).stem / "preview.m4a"
        if not p.exists():
            p = Path(job["source_path"])
        return FileResponse(p)

    # --- 热词
    @app.get("/api/hotwords")
    def api_hotwords():
        return db.list_hotwords()

    @app.post("/api/hotwords")
    async def api_hotword_add(req: Request):
        b = await req.json()
        term = (b.get("term") or "").strip()
        if not term:
            raise HTTPException(400, "term required")
        db.upsert_hotword(term, b.get("type") or "term", b.get("aliases") or [],
                          float(b.get("weight", 1.0)), b.get("note") or "")
        return {"ok": True}

    @app.delete("/api/hotwords/{term}")
    def api_hotword_del(term: str):
        db.delete_hotword(term)
        return {"ok": True}

    # --- 声纹
    @app.get("/api/voices")
    def api_voices():
        from .pipeline import cluster_channels
        import numpy as np
        rows = db.list_speakers()
        thr = float(getattr(cfg.diarization, "channel_threshold", 0.75))
        maxk = int(getattr(cfg.diarization, "max_channels", 4))
        for sp in rows:
            vps = db.voiceprint_vectors(sp["id"])
            vecs = [np.frombuffer(v["vector"], dtype=np.float32) for v in vps if v.get("vector")]
            try:
                cents, ids = cluster_channels(vecs, thr, maxk)
            except Exception:
                cents, ids = [], []
            sp["channels"] = len(cents)
            sp["channel_sizes"] = [ids.count(i) for i in range(len(cents))]
        return rows

    @app.post("/api/voices")
    async def api_voice_create(name: str = Form(""),
                               files: list[UploadFile] = File(default=[]),
                               file: UploadFile | None = File(default=None)):
        """多段声纹登记：一次提交 1–4 段音频，各自提取向量后合并入库。"""
        name = (name or "").strip()
        if not name:
            return {"ok": False, "error": "姓名为空"}
        if db.get_speaker_by_name(name):
            return {"ok": False, "error": f"{name} 已存在"}

        uploads = list(files or [])
        if file is not None:
            uploads.append(file)
        if not uploads:
            return {"ok": False, "error": "没有收到音频"}

        import soundfile as sf
        spk = SpeakerEngine(cfg.diarization.campplus_model)
        vecs: list[np.ndarray] = []
        clips: list[dict] = []
        warnings: list[str] = []

        for i, up in enumerate(uploads):
            key = f"段{i + 1}"
            tmp = Path(tempfile.mkdtemp()) / "sample"
            tmp.write_bytes(await up.read())
            wav = tmp.with_suffix(".wav")
            try:
                FFmpeg.normalize(tmp, wav)
                audio, sr = sf.read(str(wav), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                dur = len(audio) / sr
                if dur < 1.5:
                    clips.append({"key": key, "ok": False, "dur": round(dur, 1),
                                  "error": "太短（<1.5s）"})
                    continue
                vec = spk.embed(audio)
                if vec is None:
                    clips.append({"key": key, "ok": False, "dur": round(dur, 1),
                                  "error": "无法提取声纹"})
                    continue
                vecs.append(np.asarray(vec, dtype=np.float32))
                clips.append({"key": key, "ok": True, "dur": round(dur, 1)})
            except Exception as e:
                clips.append({"key": key, "ok": False, "error": f"处理失败：{e}"})
            finally:
                for f in (tmp, wav):
                    try:
                        f.unlink()
                    except Exception:
                        pass

        valid = len(vecs)
        if valid == 0:
            return {"ok": False, "error": "没有可用样本", "clips": clips, "total": len(uploads)}

        # 三句自一致性：两两余弦的均值（文档 §6.4 步骤 6）
        M = np.stack([v / (np.linalg.norm(v) or 1) for v in vecs])
        if valid >= 2:
            sims = M @ M.T
            tri = sims[np.triu_indices(valid, 1)]
            consistency = float(np.mean(tri))
            if consistency < 0.70:
                warnings.append(f"三句自一致性偏低（{consistency:.2f} < 0.70），可能有插话或环境变化，建议重录")
        else:
            consistency = None
            warnings.append("只录了 1 段，建议补录到 3 段")
        if valid < 3:
            warnings.append(f"有效样本 {valid}/3，样本越多认名越稳")

        sid = db.create_speaker(name)
        for v, c in zip(vecs, [c for c in clips if c.get("ok")]):
            db.add_voiceprint(sid, v.tobytes(), int(v.shape[0]), source="web",
                              quality=min(1.0, (c.get("dur") or 0) / 12))
        db.log(sid, "info", "voices",
               f"登记 {name}：{valid}/{len(uploads)} 段，自一致性 {consistency}")
        return {"ok": True, "id": sid, "name": name, "valid": valid, "total": len(uploads),
                "clips": clips, "consistency": round(consistency, 3) if consistency is not None else None,
                "warnings": warnings}

    @app.get("/api/voices/{sid}/samples")
    def api_voice_samples(sid: str):
        sp = db.get_speaker(sid)
        if not sp:
            raise HTTPException(404, "人物不存在")
        from .pipeline import cluster_channels
        import numpy as np
        items = db.list_voiceprints(sid)
        vps = db.voiceprint_vectors(sid)
        vecs = [np.frombuffer(v["vector"], dtype=np.float32) for v in vps if v.get("vector")]
        thr = float(getattr(cfg.diarization, "channel_threshold", 0.75))
        maxk = int(getattr(cfg.diarization, "max_channels", 4))
        try:
            cents, ids = cluster_channels(vecs, thr, maxk)
        except Exception:
            cents, ids = [], []
        # items 与 vps 同为按 id 排序，长度一致
        for v, cid in zip(items, ids):
            v["channel"] = cid + 1        # 1 开始，方便显示
        return {"speaker": sp["name"], "channels": len(cents), "items": items}

    @app.patch("/api/voices/{sid}")
    def api_voice_rename(sid: str, payload: dict):
        name = (payload.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "姓名不能为空")
        sp = db.get_speaker(sid)
        if not sp:
            raise HTTPException(404, "人物不存在")
        other = db.get_speaker_by_name(name)
        if other and other["id"] != sid:
            raise HTTPException(400, f"{name} 已存在")
        db.rename_speaker(sid, name)
        db.log(sid, "info", "voices", f"声纹改名 {sp['name']} → {name}")
        return {"ok": True, "name": name}

    @app.delete("/api/voices/{sid}/samples/{vid}")
    def api_voice_del_sample(sid: str, vid: int):
        rows = db.list_voiceprints(sid)
        if not any(int(r["id"]) == vid for r in rows):
            raise HTTPException(404, "样本不存在")
        if len(rows) <= 1:
            raise HTTPException(400, "只剩一条样本，删掉就无法认名了；可直接删除人物")
        db.delete_voiceprint(vid)
        db.log(sid, "info", "voices", f"删除样本 {vid}")
        return {"ok": True, "remaining": len(rows) - 1}

    @app.post("/api/voices/{sid}/samples")
    async def api_voice_append(sid: str, files: list[UploadFile] = File(default=[]),
                               file: UploadFile | None = File(default=None)):
        """给已有人物追加样本（可以一次多段）。"""
        sp = db.get_speaker(sid)
        if not sp:
            raise HTTPException(404, "人物不存在")
        uploads = list(files or [])
        if file is not None:
            uploads.append(file)
        if not uploads:
            raise HTTPException(400, "没有收到音频")
        import soundfile as sf
        eng = SpeakerEngine(cfg.diarization.campplus_model)
        added, skipped = 0, []
        for i, up in enumerate(uploads):
            tmp = Path(tempfile.mkdtemp()) / "sample"
            tmp.write_bytes(await up.read())
            wav = tmp.with_suffix(".wav")
            try:
                FFmpeg.normalize(tmp, wav)
                audio, sr = sf.read(str(wav), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                dur = len(audio) / sr
                if dur < 1.5:
                    skipped.append(f"第{i + 1}段太短（{dur:.1f}s）")
                    continue
                vec = eng.embed(audio)
                if vec is None:
                    skipped.append(f"第{i + 1}段无法提取声纹")
                    continue
                db.add_voiceprint(sid, np.asarray(vec, dtype=np.float32).tobytes(),
                                  int(vec.shape[0]), source="append",
                                  quality=min(1.0, dur / 12))
                added += 1
            except Exception as e:
                skipped.append(f"第{i + 1}段失败：{e}")
            finally:
                for f in (tmp, wav):
                    try:
                        f.unlink()
                    except Exception:
                        pass
        # 追加后重新算自一致性
        vecs = [np.frombuffer(v, dtype=np.float32) for v in db.speaker_vectors(sid)]
        cons = None
        if len(vecs) >= 2:
            M = np.stack([v / (np.linalg.norm(v) or 1) for v in vecs])
            sims = M @ M.T
            tri = sims[np.triu_indices(len(vecs), 1)]
            cons = round(float(np.mean(tri)), 3)
            if cons < 0.70:
                skipped.append(f"样本自一致性偏低（{cons} < 0.70），建议删掉差异大的样本")
        if added:
            db.log(sid, "info", "voices", f"{sp['name']} 追加 {added} 段样本")
        return {"ok": added > 0, "added": added, "total": db.count_voiceprints(sid),
                "consistency": cons, "warnings": skipped}

    @app.delete("/api/voices/{sid}")
    def api_voice_delete(sid: str):
        sp = db.get_speaker(sid)
        db.delete_speaker(sid)
        db.log(sid, "warning", "voices", f"删除人物 {sp['name'] if sp else sid} 及其全部样本")
        return {"ok": True}

    @app.get("/api/voices/export")
    def api_voices_export():
        return [{"name": s["name"], "samples": s["samples"]} for s in db.list_speakers()]

    @app.get("/api/jobs/{jid}/stages")
    def api_stages(jid: str):
        job = db.get_job(jid)
        if not job:
            raise HTTPException(404)
        from . import flow
        with db.conn() as c:
            evs = [dict(r) for r in c.execute(
                "SELECT stage,level,message,ts FROM events WHERE job_id=? ORDER BY id", (jid,))]
        nodes = flow.build(dict(job), evs)
        return {"nodes": nodes, "summary": flow.summary(nodes),
                "status": job["status"], "stage": job["stage"], "progress": job["progress"]}

    @app.get("/api/jobs/{jid}/events")
    async def api_events(jid: str):
        async def gen():
            last = None
            while True:
                job = db.get_job(jid)
                if not job:
                    yield "data: {}\n\n"; return
                snap = {"status": job["status"], "stage": job["stage"],
                        "progress": job["progress"], "message": job["message"]}
                if snap != last:
                    yield f"data: {json.dumps(snap, ensure_ascii=False)}\n\n"
                    last = snap
                if job["status"] in ("done", "failed", "canceled"):
                    return
                await asyncio.sleep(1.0)
        return StreamingResponse(gen(), media_type="text/event-stream")

    if getattr(cfg, "openai_api", None) and cfg.openai_api.enabled:
        app.include_router(build_router(cfg, db))

    @app.get("/api/version")
    def api_version():
        import time as _t
        mt = _code_mtime()
        return {"version": APP_VERSION,
                "started_at": _t.strftime("%Y-%m-%dT%H:%M:%S", _t.localtime(APP_START)),
                "code_mtime": _t.strftime("%Y-%m-%dT%H:%M:%S", _t.localtime(mt)) if mt else "",
                "stale": mt > APP_START + 1}

    @app.get("/api/health")
    def health():
        return {"ok": True, "jobs": len(db.list_jobs()), "voices": len(db.list_speakers())}

    return app
