"""Web 对弈前端页面(单页,canvas 渲染)。

拆为多个片段拼接,避免单个写入过大。坐标约定与后端一致:
row 0 为红方底线(显示在棋盘下方),row 9 为黑方底线(上方)。
"""

# ---- HTML 结构与样式 ----
_HTML_HEAD = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AlphaZero 中国象棋</title>
<style>
  body { margin:0; font-family: system-ui, sans-serif; background:#2b2b2b;
         color:#eee; display:flex; flex-direction:column; align-items:center; }
  h1 { font-size:18px; font-weight:600; margin:14px 0 6px; letter-spacing:1px; }
  #status { height:22px; margin:4px 0 10px; font-size:14px; color:#ffd27f; }
  canvas { background:#f0d9a8; border-radius:6px; box-shadow:0 4px 18px rgba(0,0,0,.5);
           cursor:pointer; touch-action:none; }
  #controls { margin:12px 0 20px; display:flex; gap:10px; }
  button { background:#444; color:#eee; border:1px solid #666; border-radius:5px;
           padding:7px 16px; font-size:14px; cursor:pointer; }
  button:hover { background:#555; }
  #hint { font-size:12px; color:#999; margin-bottom:18px; max-width:420px;
          text-align:center; line-height:1.5; }
</style>
</head>
<body>
<h1>AlphaZero 中国象棋</h1>
<div id="status">加载中…</div>
<canvas id="board"></canvas>
<div id="controls">
  <button onclick="newGame()">新对局</button>
  <button onclick="aiMove()">让 AI 走</button>
</div>
<div id="hint">点击己方棋子选中,绿点为可落点,再点目标处走子。AI 默认执黑,
你走后点「让 AI 走」应招(或走子后自动应招)。</div>
"""

# ---- JS:常量与渲染 ----
_JS_RENDER = """
<script>
const ROWS = 10, COLS = 9;
const M = 36;            // 边距
const G = 56;            // 格间距
const W = M*2 + G*(COLS-1);
const H = M*2 + G*(ROWS-1);
const canvas = document.getElementById('board');
canvas.width = W; canvas.height = H;
const ctx = canvas.getContext('2d');

let state = null;        // {grid,to_move,result,in_check}
let selected = null;     // [r,c]
let targets = [];        // [[r,c],...]
let lastMove = null;     // [[r,c],[r,c]]

// 棋子中文名:索引为绝对值,正红负黑
const NAMES = {1:['帅','将'],2:['仕','士'],3:['相','象'],
               4:['马','马'],5:['车','车'],6:['炮','炮'],7:['兵','卒']};

// 棋盘坐标 (row,col) -> 画布像素。row 0 在底部。
function px(r,c){ return [M + c*G, M + (ROWS-1-r)*G]; }
// 画布像素 -> 最近的棋盘坐标
function toCell(x,y){
  const c = Math.round((x-M)/G);
  const r = ROWS-1 - Math.round((y-M)/G);
  if(r<0||r>=ROWS||c<0||c>=COLS) return null;
  return [r,c];
}

function drawBoard(){
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle = '#5a3a18'; ctx.lineWidth = 1.4;
  // 横线
  for(let r=0;r<ROWS;r++){
    const [x0,y0]=px(r,0), [x1,_]=px(r,COLS-1);
    ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y0); ctx.stroke();
  }
  // 竖线(中间断开为楚河汉界,除最外两条)
  for(let c=0;c<COLS;c++){
    const [x,_]=px(0,c);
    if(c===0||c===COLS-1){
      const [_a,yT]=px(ROWS-1,c), [_b,yB]=px(0,c);
      ctx.beginPath(); ctx.moveTo(x,yT); ctx.lineTo(x,yB); ctx.stroke();
    } else {
      const [_a,y4]=px(4,c), [_b,y0]=px(0,c);
      ctx.beginPath(); ctx.moveTo(x,y0); ctx.lineTo(x,y4); ctx.stroke();
      const [_c,y9]=px(9,c), [_d,y5]=px(5,c);
      ctx.beginPath(); ctx.moveTo(x,y9); ctx.lineTo(x,y5); ctx.stroke();
    }
  }
  // 九宫斜线
  drawPalace(0); drawPalace(7);
  // 楚河汉界文字
  ctx.fillStyle='#5a3a18'; ctx.font='20px serif'; ctx.textAlign='center';
  const [hx,hy]=px(4,4); // 河中央附近
  const midY = (px(4,0)[1]+px(5,0)[1])/2;
  ctx.fillText('楚 河', M+G*1.5, midY+7);
  ctx.fillText('汉 界', M+G*6.5, midY+7);
}

function drawPalace(baseRow){
  const corners = [[baseRow,3],[baseRow,5],[baseRow+2,3],[baseRow+2,5]];
  const [a,b,c,d] = corners.map(([r,cc])=>px(r,cc));
  ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(d[0],d[1]); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(b[0],b[1]); ctx.lineTo(c[0],c[1]); ctx.stroke();
}

function drawPieces(){
  if(!state) return;
  for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++){
    const p = state.grid[r][c];
    if(p===0) continue;
    const [x,y]=px(r,c);
    const red = p>0;
    const name = NAMES[Math.abs(p)][red?0:1];
    // 棋子圆盘
    ctx.beginPath(); ctx.arc(x,y,G*0.42,0,Math.PI*2);
    ctx.fillStyle = '#f7ecd0'; ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = red ? '#c0392b' : '#2c3e50'; ctx.stroke();
    ctx.fillStyle = red ? '#c0392b' : '#2c3e50';
    ctx.font = 'bold 28px serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(name, x, y+1);
  }
}

function drawOverlays(){
  // 上一步走子标记
  if(lastMove){
    for(const [r,c] of lastMove){
      const [x,y]=px(r,c);
      ctx.strokeStyle='#3498db'; ctx.lineWidth=2.5;
      ctx.strokeRect(x-G*0.46,y-G*0.46,G*0.92,G*0.92);
    }
  }
  // 选中棋子
  if(selected){
    const [x,y]=px(selected[0],selected[1]);
    ctx.strokeStyle='#27ae60'; ctx.lineWidth=3;
    ctx.beginPath(); ctx.arc(x,y,G*0.46,0,Math.PI*2); ctx.stroke();
  }
  // 合法落点
  for(const [r,c] of targets){
    const [x,y]=px(r,c);
    const occupied = state.grid[r][c]!==0;
    ctx.fillStyle = occupied ? 'rgba(231,76,60,.55)' : 'rgba(39,174,96,.65)';
    ctx.beginPath(); ctx.arc(x,y, occupied?G*0.44:G*0.16, 0, Math.PI*2); ctx.fill();
  }
}

function render(){
  drawBoard(); drawPieces(); drawOverlays();
}
</script>
"""

# ---- JS:交互与 API ----
_JS_LOGIC = """
<script>
const SIDE_NAME = {'1':'红','-1':'黑'};
const RESULT_TEXT = {
  'ongoing':'', 'red_win':'红方胜', 'black_win':'黑方胜', 'draw':'和棋'
};
let aiSide = -1;        // 由后端决定,这里默认黑;通过 status 推断不必精确
let busy = false;

async function api(path, body){
  const r = await fetch(path, {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: body ? JSON.stringify(body) : null,
  });
  return r.json();
}

function setStatus(){
  const el = document.getElementById('status');
  if(!state){ el.textContent=''; return; }
  const res = RESULT_TEXT[state.result];
  if(res){ el.textContent = '对局结束:' + res; return; }
  let s = '轮到 ' + SIDE_NAME[state.to_move] + ' 方';
  if(state.in_check) s += ' · 将军!';
  el.textContent = s;
}

function applyState(s){
  state = s; setStatus(); render();
}

async function newGame(){
  busy = true;
  selected = null; targets = []; lastMove = null;
  applyState(await api('/api/new'));
  busy = false;
}

async function aiMove(){
  if(busy || !state || state.result!=='ongoing') return;
  busy = true;
  setStatusBusy('AI 思考中…');
  const s = await api('/api/ai');
  if(s.ai_move) lastMove = s.ai_move;
  selected = null; targets = [];
  applyState(s);
  busy = false;
}

function setStatusBusy(t){ document.getElementById('status').textContent = t; }

async function onClick(ev){
  if(busy || !state || state.result!=='ongoing') return;
  const rect = canvas.getBoundingClientRect();
  const x = (ev.clientX-rect.left) * (canvas.width/rect.width);
  const y = (ev.clientY-rect.top) * (canvas.height/rect.height);
  const cell = toCell(x,y);
  if(!cell) return;
  const [r,c] = cell;
  const piece = state.grid[r][c];

  // 若点到合法落点 -> 走子
  if(selected && targets.some(t=>t[0]===r&&t[1]===c)){
    busy = true;
    const s = await api('/api/move', {from:selected, to:[r,c]});
    if(!s.error){
      lastMove = [selected.slice(), [r,c]];
      selected = null; targets = [];
      applyState(s);
      busy = false;
      // 轮到 AI 则自动应招
      if(state.result==='ongoing') await aiMove();
    } else { busy = false; }
    return;
  }

  // 点己方棋子 -> 选中并取合法落点
  if(piece!==0 && Math.sign(piece)===state.to_move){
    selected = [r,c];
    const res = await api('/api/legal', {from:[r,c]});
    targets = res.targets || [];
    render();
  } else {
    selected = null; targets = []; render();
  }
}

canvas.addEventListener('click', onClick);
newGame();
</script>
</body>
</html>
"""

PAGE_HTML = _HTML_HEAD + _JS_RENDER + _JS_LOGIC


