'use strict';
// The state filter of skew.html: hides the rows of the table whose state differs from the one selected.
function filter(){
 const state=document.getElementById('state').value;
 for(const row of document.querySelectorAll('#skew-orders tbody tr'))row.classList.toggle('hidden',!!state&&row.dataset.state!==state);
}
document.getElementById('state').addEventListener('change',filter);
filter();
