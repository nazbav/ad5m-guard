"""Панель управления парком принтеров в браузере (Flask).

    /                       страница: карточки принтеров с живыми кадрами и рамками, журнал
    /api/state              состояние парка (JSON)
    /api/frame/<имя>.jpg    последний кадр принтера с рамками
    POST /api/printer/<имя>/<pause|resume|cancel>   ручное управление

Если задан токен (DASHBOARD_TOKEN), управление требует его в заголовке X-Token
или параметре ?token= — иначе любой в сети мог бы останавливать печать.
"""
from __future__ import annotations

import hmac

from flask import Flask, Response, abort, jsonify, request

PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Парк AD5M</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--fg:#1c1d20;--muted:#6b6f76;--line:#e1e3e8;--ok:#1f8a4c;--warn:#b86e00;--bad:#c4302b;--idle:#5b6472;--accent:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#121316;--card:#1b1d21;--fg:#e8e9ec;--muted:#9aa0a8;--line:#2c2f35;--ok:#3cc47a;--warn:#e5a13a;--bad:#ef5a55;--idle:#8b94a3;--accent:#6ea0ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 "Segoe UI",system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:10px 16px;background:var(--card);border-bottom:1px solid var(--line)}
h1{font-size:17px;margin:0 8px 0 0}.pill{padding:2px 10px;border-radius:999px;font-size:13px;border:1px solid var(--line)}
.meta{color:var(--muted);font-size:12px;margin-left:auto}
main{display:grid;grid-template-columns:1fr 340px;gap:16px;padding:16px}
@media (max-width:1000px){main{grid-template-columns:1fr}}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;align-content:start}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column}
.card.alert{border-color:var(--bad);box-shadow:0 0 0 2px color-mix(in srgb,var(--bad) 35%,transparent)}
.top{display:flex;align-items:center;gap:8px;padding:8px 10px}.name{font-weight:600;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.badge{font-size:12px;padding:1px 8px;border-radius:6px;color:#fff;background:var(--idle)}
.b-printing{background:var(--ok)}.b-paused,.b-pausing{background:var(--warn)}.b-error,.b-cancel{background:var(--bad)}
.shot{position:relative;aspect-ratio:4/3;background:#000}.shot img{width:100%;height:100%;object-fit:cover;display:block}
.shot .none{position:absolute;inset:0;display:grid;place-items:center;color:#888;font-size:13px}
.bar{height:4px;background:var(--line)}.bar>i{display:block;height:100%;background:var(--accent)}
.info{padding:8px 10px;display:flex;flex-direction:column;gap:6px;font-size:13px}
.chips{display:flex;flex-wrap:wrap;gap:4px}.chip{font-size:12px;padding:0 7px;border-radius:5px;background:color-mix(in srgb,var(--bad) 18%,transparent);color:var(--bad)}.chip.note{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}.chip.muted{background:color-mix(in srgb,var(--idle) 16%,transparent);color:var(--muted)}
.muted{color:var(--muted)}.err{color:var(--bad)}
.btns{display:flex;gap:6px;padding:0 10px 10px}.btns button{flex:1;padding:6px;border-radius:6px;border:1px solid var(--line);background:transparent;color:var(--fg);cursor:pointer;font:inherit}
.btns button:hover{border-color:var(--accent)}.btns button.danger:hover{border-color:var(--bad);color:var(--bad)}
aside{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;max-height:calc(100vh - 90px);overflow:auto;position:sticky;top:70px}
aside h2{font-size:14px;margin:0 0 8px}.ev{padding:5px 0;border-bottom:1px solid var(--line);font-size:12.5px}
.ev b{font-weight:600}.k-alert,.k-stopped,.k-error{color:var(--bad)}.k-manual{color:var(--accent)}
</style></head><body>
<header><h1>Парк AD5M</h1><span id="counts"></span><span class="meta" id="meta"></span></header>
<main><section class="grid" id="grid"></section><aside><h2>Журнал</h2><div id="events"></div></aside></main>
<script>
const LABEL={printing:"печатает",paused:"пауза",pausing:"ставится на паузу",ready:"готов",completed:"закончил",cancel:"отменена",error:"ошибка",busy:"занят",heating:"нагрев",unknown:"—"};
const token=new URLSearchParams(location.search).get("token")||"";
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
async function act(name,a){
  if(a==="cancel"&&!confirm(`Отменить печать на ${name}?`))return;
  const r=await fetch(`/api/printer/${encodeURIComponent(name)}/${a}?token=${encodeURIComponent(token)}`,{method:"POST"});
  const j=await r.json().catch(()=>({message:r.statusText}));if(!r.ok||!j.ok)alert(`${name}: ${j.message||"ошибка"}`);refresh();
}
let S={muted:[],notify_only:[]};
function card(p){
  const st=p.error?"error":p.status, prog=p.progress!=null?Math.round(p.progress*100):null;
  const serious=p.problems.filter(x=>!S.muted.includes(x)&&!S.notify_only.includes(x));
  const alertNow=serious.length>0||(p.status==="paused"&&p.last_alert);
  const cls=x=>S.muted.includes(x)?"chip muted":S.notify_only.includes(x)?"chip note":"chip";
  return `<div class="card${alertNow?" alert":""}">
   <div class="top"><span class="name" title="${esc(p.host)}">${esc(p.name)}</span><span class="badge b-${esc(st)}">${esc(LABEL[st]||st)}</span></div>
   <div class="shot">${p.has_frame?`<img alt="" src="/api/frame/${encodeURIComponent(p.name)}.jpg?t=${encodeURIComponent(p.frame_at)}">`:`<div class="none">нет кадра</div>`}</div>
   <div class="bar"><i style="width:${prog??0}%"></i></div>
   <div class="info">
     ${p.error?`<div class="err">${esc(p.error)}</div>`:""}
     ${p.problems.length?`<div class="chips">${p.problems.map(x=>`<span class="${cls(x)}" title="${S.muted.includes(x)?"тревога по этому отключена":S.notify_only.includes(x)?"только уведомление":"остановит печать"}">${esc(x)}</span>`).join("")}</div>`:`<div class="muted">проблем на кадре нет</div>`}
     <div class="muted">${prog!=null?`прогресс ${prog}% · `:""}${p.last_alert?`тревога: ${esc(p.last_alert)} (${esc(p.last_alert_at.slice(11))})`:"тревог не было"}</div>
   </div>
   <div class="btns"><button onclick="act('${esc(p.name)}','pause')">Пауза</button><button onclick="act('${esc(p.name)}','resume')">Продолжить</button><button class="danger" onclick="act('${esc(p.name)}','cancel')">Отменить</button></div>
  </div>`;
}
async function refresh(){
  try{
    const s=await (await fetch("/api/state")).json();S=s;
    document.getElementById("grid").innerHTML=s.printers.map(card).join("");
    document.getElementById("counts").innerHTML=Object.entries(s.counts).map(([k,v])=>`<span class="pill">${esc(LABEL[k]||k)}: ${v}</span>`).join(" ");
    document.getElementById("meta").textContent=`цикл ${s.cycle} · опрос раз в ${s.interval_s} с · при тревоге: ${s.action} · порог ${s.defect_conf}`;
    document.getElementById("events").innerHTML=s.events.map(e=>`<div class="ev"><span class="muted">${esc(e.when.slice(11))}</span> <b>${esc(e.printer)}</b> <span class="k-${esc(e.kind)}">${esc(e.detail)}</span></div>`).join("")||`<div class="muted">пока пусто</div>`;
  }catch(e){document.getElementById("meta").textContent="нет связи с сервисом"}
}
refresh();setInterval(refresh,2000);
</script></body></html>"""


def create_app(monitor, token: str | None = None) -> Flask:
    app = Flask(__name__)

    def authorized() -> bool:
        if not token:
            return True
        given = request.headers.get("X-Token") or request.args.get("token") or ""
        return hmac.compare_digest(given, token)

    @app.get("/")
    def index():
        return Response(PAGE, mimetype="text/html")

    @app.get("/api/state")
    def state():
        return jsonify(monitor.state())

    @app.get("/api/frame/<name>.jpg")
    def frame(name: str):
        jpg = monitor.frame(name)
        if jpg is None:
            abort(404)
        return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/printer/<name>/<action>")
    def control(name: str, action: str):
        if not authorized():
            return jsonify(ok=False, message="нужен токен панели"), 403
        ok, message = monitor.control(name, action)
        return jsonify(ok=ok, message=message), (200 if ok else 400)

    return app
