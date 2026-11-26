'use strict';
// The New results table sorts by a column when its header is clicked; a second click reverses the order.
for(const table of document.querySelectorAll('table.results')){
 const body=table.tBodies[0];[...body.rows].forEach((row,i)=>{row.dataset.i=i;});
 const heads=[...table.tHead.rows[0].cells];
 heads.forEach((th,col)=>{
  const button=document.createElement('button');button.type='button';button.textContent=th.textContent;th.textContent='';th.append(button);
  button.addEventListener('click',()=>{
   const dir=th.getAttribute('aria-sort')==='ascending'?-1:1;
   for(const other of heads)other.removeAttribute('aria-sort');
   th.setAttribute('aria-sort',dir>0?'ascending':'descending');
   const key=row=>{const cell=row.cells[col];return cell.dataset.sort??cell.textContent.trim();};
   const rows=[...body.rows].sort((a,b)=>{const x=key(a),y=key(b),nx=Number(x),ny=Number(y);
    const d=x!==''&&y!==''&&!Number.isNaN(nx)&&!Number.isNaN(ny)?nx-ny:x.localeCompare(y,undefined,{numeric:true});
    return d*dir||a.dataset.i-b.dataset.i;});
   body.append(...rows);
  });
 });
}
function filter(){
 const track=document.getElementById('track').value,state=document.getElementById('status').value,max=Number(document.getElementById('max-order').value),origin=document.getElementById('origin').value;
 for(const row of document.querySelectorAll('#grid tbody tr')){
  let visible=false;
  for(const td of row.querySelectorAll('td[data-track]')){
   const column=!track||td.dataset.track===track;
   const matches=(!state||td.dataset.status===state||td.dataset.artifact===state)&&(!origin||td.dataset.origins.split(' ').includes(origin));
   td.classList.toggle('hidden',!column);
   if(td.classList.contains('blank'))continue;
   td.style.visibility=matches?'':'hidden';visible ||=column&&matches;
  }
  row.classList.toggle('hidden',!visible||Number(row.dataset.n)>max);
 }
 for(const th of document.querySelectorAll('#grid th[data-track]'))th.classList.toggle('hidden',!!track&&th.dataset.track!==track);
 // A group header would span columns the filter hides.
 for(const th of document.querySelectorAll('#grid th.group'))th.classList.toggle('hidden',!!track);
}
for(const id of ['track','status','origin','max-order'])document.getElementById(id).addEventListener('change',filter);
filter();
// Older links addressed a cell as #cell-<n>-<track>; each cell now has its own page.
function openLinkedCell(){const match=location.hash.match(/^#cell-(\d+)-([a-z-]+)$/);if(match)location.replace(`cells/n${match[1]}-${match[2]}.html`);}
window.addEventListener('hashchange',openLinkedCell);
openLinkedCell();
