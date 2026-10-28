'use strict';
// Draws the four defining rows of a quadruple and the Goethals-Seidel array
// they generate, for every figure.quad on the page. This is an illustration
// computed from the submitted rows; the verifier result is the evidence.
// Hue follows the symmetry of the block: skew blocks green, symmetric blue.
// Light is +1 and dark is -1 in both hues.
const PALETTE={k:[[159,211,181,255],[20,38,29,255]],s:[[169,199,232,255],[20,32,44,255]]};
function paint(canvas,width,height,value,kind){
 if(!canvas)return;
 canvas.width=width;canvas.height=height;
 const ctx=canvas.getContext('2d'),img=ctx.createImageData(width,height),px=img.data;
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  const tones=PALETTE[kind(y,x)]||PALETTE.k,c=value(y,x)>0?tones[0]:tones[1],k=4*(y*width+x);
  px[k]=c[0];px[k+1]=c[1];px[k+2]=c[2];px[k+3]=c[3];
 }
 ctx.putImageData(img,0,0);
}
function draw(figure){
 const quad=JSON.parse(figure.querySelector('script[type="application/json"]').textContent);
 const n=quad.n,rows=quad.rows,[A,B,C,D]=rows,pattern=quad.pattern||'kkkk';
 // Circulant block X: entry (i, j) is x[(j - i) mod n].
 const circ=(x,i,j)=>x[((j-i)%n+n)%n];
 // X R reverses the columns of X; R X reverses its rows; both stay explicit so
 // the layout below reads like the array it draws.
 const xR=(x,i,j)=>circ(x,i,n-1-j);
 const Rx=(x,i,j)=>circ(x,n-1-i,j);
 const layout=[
  [(i,j)=>circ(A,i,j),(i,j)=>xR(B,i,j),(i,j)=>xR(C,i,j),(i,j)=>xR(D,i,j)],
  [(i,j)=>-xR(B,i,j),(i,j)=>circ(A,i,j),(i,j)=>-Rx(D,i,j),(i,j)=>Rx(C,i,j)],
  [(i,j)=>-xR(C,i,j),(i,j)=>Rx(D,i,j),(i,j)=>circ(A,i,j),(i,j)=>-Rx(B,i,j)],
  [(i,j)=>-xR(D,i,j),(i,j)=>-Rx(C,i,j),(i,j)=>Rx(B,i,j),(i,j)=>circ(A,i,j)],
 ];
 // Block (bi, bj) of the array is the circulant of row bi XOR bj, up to sign and R.
 paint(figure.querySelector('canvas.rows'),n,4,(y,x)=>rows[y][x],(y,x)=>pattern[y]);
 paint(figure.querySelector('canvas.array'),4*n,4*n,(y,x)=>layout[Math.floor(y/n)][Math.floor(x/n)](y%n,x%n),(y,x)=>pattern[Math.floor(y/n)^Math.floor(x/n)]);
}
document.querySelectorAll('figure.quad').forEach(draw);
