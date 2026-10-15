'use strict';
const data=JSON.parse(document.getElementById('board-data').textContent);
const byCell=new Map(data.cells.map(c=>[`${c.n}:${c.stratum}`,c]));
const panel=document.getElementById('cell-detail');
function el(tag,text){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;}
function showCell(key){
 const c=byCell.get(key);if(!c)return;
 panel.replaceChildren(el('h3',`n = ${c.n} · ${c.stratum}`));
 panel.append(el('p',`${c.status} · ${c.artifact_status}${c.tier?' · '+c.tier:''}`));
 if(c.note)panel.append(el('p',c.note));
 const ids=[...c.quadruples,...c.evidence];
 if(ids.length){const list=el('ul');for(const id of ids){const li=el('li');const a=el('a',id);a.href=`entries/${id}.html`;li.append(a);list.append(li);}panel.append(list);}
 for(const source of c.sources){const p=el('p');const a=el('a',source.source.title);a.href=source.source.url;a.rel='noopener';p.append(a);panel.append(p);}
 if(c.status==='unresolved')panel.append(el('p','Unresolved in the sourced baseline. Review the literature before claiming novelty.'));
 if(c.evidence.length)panel.append(el('p','Open the evidence records for scope, coverage, gaps, and the checks that actually ran.'));
 panel.hidden=false;panel.scrollIntoView({block:'nearest'});
 history.replaceState(null,'',`#cell-${c.n}-${c.stratum}`);
}
function filter(){
 const track=document.getElementById('track').value,state=document.getElementById('status').value,max=Number(document.getElementById('max-order').value);
 for(const row of document.querySelectorAll('#grid tbody tr')){
  let visible=false;
  for(const td of row.querySelectorAll('td[data-track]')){
   const column=!track||td.dataset.track===track;
   const matches=!state||td.dataset.status===state||td.dataset.artifact===state;
   td.classList.toggle('hidden',!column);
   td.style.visibility=matches?'':'hidden';visible ||=column&&matches;
  }
  row.classList.toggle('hidden',!visible||Number(row.dataset.n)>max);
 }
 for(const th of document.querySelectorAll('#grid th[data-track]'))th.classList.toggle('hidden',!!track&&th.dataset.track!==track);
}
document.querySelectorAll('[data-cell]').forEach(b=>b.addEventListener('click',()=>showCell(b.dataset.cell)));
for(const id of ['track','status','max-order'])document.getElementById(id).addEventListener('change',filter);
const match=location.hash.match(/^#cell-(\d+)-(.+)$/);if(match)showCell(`${match[1]}:${match[2]}`);
