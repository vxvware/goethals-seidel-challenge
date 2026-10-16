'use strict';
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
for(const id of ['track','status','origin','max-order'])document.getElementById(id).addEventListener('change',filter);
// Older links addressed a cell as #cell-<n>-<track>; each cell now has its own page.
function openLinkedCell(){const match=location.hash.match(/^#cell-(\d+)-([a-z-]+)$/);if(match)location.replace(`cells/n${match[1]}-${match[2]}.html`);}
window.addEventListener('hashchange',openLinkedCell);
openLinkedCell();
