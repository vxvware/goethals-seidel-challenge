'use strict';
// Draws the four defining rows and the Goethals-Seidel array they generate.
// This is an illustration computed from the submitted rows; the verifier
// result on the page is the evidence, not this drawing.
const quad=JSON.parse(document.getElementById('quadruple-data').textContent);
const n=quad.n,rows=quad.rows;
const PLUS=[32,88,61,255],MINUS=[233,236,231,255];
function paint(canvas,width,height,value){
 canvas.width=width;canvas.height=height;
 const ctx=canvas.getContext('2d'),img=ctx.createImageData(width,height),px=img.data;
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  const c=value(y,x)>0?PLUS:MINUS,k=4*(y*width+x);
  px[k]=c[0];px[k+1]=c[1];px[k+2]=c[2];px[k+3]=c[3];
 }
 ctx.putImageData(img,0,0);
}
// Circulant block X: entry (i, j) is x[(j - i) mod n].
const circ=(x,i,j)=>x[((j-i)%n+n)%n];
// X R reverses the columns of X; R X reverses its rows; both stay explicit so
// the layout below reads like the array it draws.
const xR=(x,i,j)=>circ(x,i,n-1-j);
const Rx=(x,i,j)=>circ(x,n-1-i,j);
const [A,B,C,D]=rows;
const layout=[
 [(i,j)=>circ(A,i,j),(i,j)=>xR(B,i,j),(i,j)=>xR(C,i,j),(i,j)=>xR(D,i,j)],
 [(i,j)=>-xR(B,i,j),(i,j)=>circ(A,i,j),(i,j)=>-Rx(D,i,j),(i,j)=>Rx(C,i,j)],
 [(i,j)=>-xR(C,i,j),(i,j)=>Rx(D,i,j),(i,j)=>circ(A,i,j),(i,j)=>-Rx(B,i,j)],
 [(i,j)=>-xR(D,i,j),(i,j)=>-Rx(C,i,j),(i,j)=>Rx(B,i,j),(i,j)=>circ(A,i,j)],
];
paint(document.getElementById('rows-picture'),n,4,(y,x)=>rows[y][x]);
paint(document.getElementById('array-picture'),4*n,4*n,(y,x)=>layout[Math.floor(y/n)][Math.floor(x/n)](y%n,x%n));
