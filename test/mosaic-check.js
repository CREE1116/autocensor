const sharp=require('sharp');
const {censor}=require('../electron/censor');
(async()=>{
const W=1200,H=1200;
const noise=Buffer.alloc(W*H*3);
for(let i=0;i<noise.length;i++) noise[i]=(i*7919)%256;   // high-frequency pattern
const img=await sharp(noise,{raw:{width:W,height:H,channels:3}}).png().toBuffer();
const mask=new Uint8Array(W*H); for(let y=0;y<600;y++)for(let x=0;x<600;x++)mask[y*W+x]=255;
const det={width:W,height:H,detections:[],mask};
const r=await censor(img,det,{mode:'mosaic',shape:'contour',dilateRadius:0,featherRadius:0,strength:1});
const {data}=await sharp(r.buffer).raw().toBuffer({resolveWithObject:true});
const at=(x,y)=>data[(y*W+x)*3];
// block = max(4, ceil(1200/100)) = 12 -> pixels inside one block must be identical
const a=[at(100,100),at(101,100),at(102,100),at(103,100)];
const orig=[noise[(100*W+100)*3],noise[(100*W+101)*3],noise[(100*W+102)*3]];
console.log('orig adjacent:',orig,'\nmosaic adjacent:',a);
const uniform=new Set(a).size===1;
console.log(uniform?'PASS mosaic applied (block uniform)':'FAIL mosaic not applied');
process.exit(uniform?0:1);
})().catch(e=>{console.error(e);process.exit(1)});
