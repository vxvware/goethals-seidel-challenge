'use strict';
function filter(){
 const track=document.getElementById('track').value,state=document.getElementById('status').value,max=Number(document.getElementById('max-order').value),origin=document.getElementById('origin').value;
 // Rows the board has no cells for show only with the last option of Through n, the whole range.
 const through=document.getElementById('max-order'),every=through.selectedIndex===through.options.length-1;
 for(const row of document.querySelectorAll('#grid tbody tr')){
  let visible=false;
  for(const td of row.querySelectorAll('td[data-track]')){
   const column=!track||td.dataset.track===track;
   const matches=(!state||td.dataset.status===state||td.dataset.artifact===state)&&(!origin||td.dataset.origins.split(' ').includes(origin));
   td.classList.toggle('hidden',!column);
   if(td.classList.contains('blank'))continue;
   td.style.visibility=matches?'':'hidden';visible ||=column&&matches;
  }
  row.classList.toggle('hidden',!visible||Number(row.dataset.n)>max||(!every&&row.hasAttribute('data-extra')));
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
