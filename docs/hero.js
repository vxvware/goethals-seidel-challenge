'use strict';
// The front-page animation. It cycles through a few verified quadruples from
// the board: prints the four rows, fills in the Goethals-Seidel array they
// generate (gsArray from picture.js), then checks H H^T = 4n I in the browser
// and reports what the check returned.
(()=>{
 const fig=document.querySelector('figure.hero-anim');
 if(!fig)return;
 const quads=JSON.parse(fig.querySelector('script[type="application/json"]').textContent);
 const canvas=fig.querySelector('canvas'),log=fig.querySelector('pre'),link=fig.querySelector('a');
 const still=matchMedia('(prefers-reduced-motion: reduce)').matches;
 const sign=row=>row.map(v=>v>0?'+':'-').join('');
 const hadamard=(H,m)=>{
  for(let a=0;a<m;a++)for(let b=a;b<m;b++){
   let dot=0;
   for(let k=0;k<m;k++)dot+=H.value(a,k)*H.value(b,k);
   if(dot!==(a===b?m:0))return false;
  }
  return true;
 };
 function lines(q){
  const out=[`> ${q.label}, n = ${q.n}`];
  'ABCD'.split('').forEach((name,i)=>out.push(`  ${name} ${q.pattern[i]} ${sign(q.rows[i])}`));
  out.push(`> array H, order ${4*q.n}`);
  return out;
 }
 function frame(q,H,cells){
  // Paint the first `cells` entries of H in reading order, the rest blank.
  const m=4*q.n,ctx=canvas.getContext('2d'),img=ctx.createImageData(m,m),px=img.data;
  for(let y=0;y<m;y++)for(let x=0;x<m;x++){
   const k=y*m+x,o=4*k;
   let c=[6,17,11,255];
   if(k<cells){const tones=PALETTE[H.kind(y,x)]||PLAIN;c=H.value(y,x)>0?tones[0]:tones[1];}
   else if(k<cells+m)c=[61,255,124,255];
   px[o]=c[0];px[o+1]=c[1];px[o+2]=c[2];px[o+3]=c[3];
  }
  ctx.putImageData(img,0,0);
 }
 function verdict(q,H){
  const m=4*q.n;
  return hadamard(H,m)?`> H Hᵀ = ${m} I: checked in this browser`:`> H Hᵀ = ${m} I: CHECK FAILED`;
 }
 function show(index){
  const q=quads[index%quads.length],H=gsArray(q),m=4*q.n,text=lines(q).join('\n');
  canvas.width=m;canvas.height=m;
  link.href=`entries/${q.id}.html`;link.textContent=q.id;
  if(still){log.textContent=text+'\n'+verdict(q,H);frame(q,H,m*m);return;}
  const TYPE=900,FILL=2600,HOLD=3800;
  let start=null;
  frame(q,H,0);
  function step(now){
   if(start===null)start=now;
   const t=now-start;
   if(t<TYPE){
    log.textContent=text.slice(0,Math.ceil(text.length*t/TYPE))+'█';
   }else if(t<TYPE+FILL){
    log.textContent=text;
    frame(q,H,Math.floor(m*m*(t-TYPE)/FILL));
   }else if(t<TYPE+FILL+HOLD){
    if(!step.done){frame(q,H,m*m);log.textContent=text+'\n'+verdict(q,H);step.done=true;}
   }else{show(index+1);return;}
   requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
 }
 show(0);
})();
