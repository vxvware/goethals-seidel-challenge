'use strict';
const data=JSON.parse(document.getElementById('board-data').textContent);
const byCell=new Map(data.cells.map(c=>[`${c.n}:${c.stratum}`,c]));
const panel=document.getElementById('cell-detail');
function el(tag,text){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;}
function showCell(key){
 const c=byCell.get(key);if(!c)return;
 panel.replaceChildren(el('h3',`n = ${c.n} · ${data.labels[c.stratum]||c.stratum}`));
 panel.append(el('p',`${c.status} · ${c.artifact_status}${c.tier?' · '+c.tier:''}`));
 if(c.note)panel.append(el('p',c.note));
 for(const credit of c.contributions){
  const block=el('article');block.className='credit';
  const badge=el('span',credit.label);badge.className=`origin ${credit.relation}`;block.append(badge);
  block.append(el('h4',credit.title));
  if(credit.contributors.length){const names=el('p');names.className='byline';for(const [i,person] of credit.contributors.entries()){
   if(i)names.append(document.createTextNode(', '));
   const name=el(person.github?'a':'span',person.name);if(person.github)name.href=`https://github.com/${person.github}`;names.append(name);
  }block.append(names);}
  block.append(el('p',`${credit.publication_status==='unpublished'?'Unpublished participant work':'Published result'} · ${credit.summary}`));panel.append(block);
 }
 const ids=[...c.quadruples,...c.evidence,...c.censuses];
 if(ids.length){const list=el('ul');for(const id of ids){const li=el('li');const a=el('a',id);a.href=`entries/${id}.html`;li.append(a);list.append(li);}panel.append(list);}
 for(const source of c.sources){const p=el('p');const a=el('a',source.source.title);a.href=source.source.url;a.rel='noopener';p.append(a);panel.append(p);}
 for(const claim of c.contradicted){const p=el('p');p.className='contradiction';p.append(document.createTextNode('Exactly verified rows on this board contradict a published nonexistence claim: '));const a=el('a',claim.source.title);a.href=claim.source.url;a.rel='noopener';p.append(a);panel.append(p);}
 if(c.status==='unresolved')panel.append(el('p','Unresolved in the sourced baseline. Review the literature before claiming novelty.'));
 if(c.evidence.length)panel.append(el('p','Open the evidence records for scope, coverage, gaps, and the checks that actually ran.'));
 panel.hidden=false;panel.scrollIntoView({block:'nearest'});
 history.replaceState(null,'',`#cell-${c.n}-${c.stratum}`);
}
function filter(){
 const track=document.getElementById('track').value,state=document.getElementById('status').value,max=Number(document.getElementById('max-order').value),origin=document.getElementById('origin').value;
 for(const row of document.querySelectorAll('#grid tbody tr')){
  let visible=false;
  for(const td of row.querySelectorAll('td[data-track]')){
   const column=!track||td.dataset.track===track;
   const matches=(!state||td.dataset.status===state||td.dataset.artifact===state)&&(!origin||td.dataset.origins.split(' ').includes(origin));
   td.classList.toggle('hidden',!column);
   td.style.visibility=matches?'':'hidden';visible ||=column&&matches;
  }
  row.classList.toggle('hidden',!visible||Number(row.dataset.n)>max);
 }
 for(const th of document.querySelectorAll('#grid th[data-track]'))th.classList.toggle('hidden',!!track&&th.dataset.track!==track);
}
document.querySelectorAll('[data-cell]').forEach(b=>b.addEventListener('click',()=>showCell(b.dataset.cell)));
for(const id of ['track','status','origin','max-order'])document.getElementById(id).addEventListener('change',filter);
function openLinkedCell(){const match=location.hash.match(/^#cell-(\d+)-(.+)$/);if(match)showCell(`${match[1]}:${match[2]}`);}
window.addEventListener('hashchange',openLinkedCell);
openLinkedCell();
