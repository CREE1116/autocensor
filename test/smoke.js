const sharp=require('sharp');
(async()=>{
const {detect}=require('/Users/leejongmin/code/autocensor/electron/detector');
const {censor}=require('/Users/leejongmin/code/autocensor/electron/censor');
const {defaultLabelConfig}=require('/Users/leejongmin/code/autocensor/electron/labels');
const img=await sharp({create:{width:1800,height:2400,channels:3,background:{r:220,g:200,b:190}}}).png().toBuffer();
for (const model of ['anime-nano','anime-medium']){
  const t=Date.now();
  const d=await detect(img,{model,labelConfig:defaultLabelConfig(),tiling:'auto',
    onProgress:e=>process.stdout.write(`\r${model} tile ${e.done}/${e.total}   `)});
  console.log(`\n${model}: ${d.detections.length} dets, ${((Date.now()-t)/1000).toFixed(1)}s, mask ${d.width}x${d.height}`);
}
// mosaic + blur path
const d=await detect(img,{model:'anime-nano',labelConfig:defaultLabelConfig(),tiling:'never'});
d.mask.fill(0); for(let y=100;y<300;y++)for(let x=100;x<300;x++)d.mask[y*1800+x]=255;
for (const mode of ['white','black','mosaic','blur']){
  const r=await censor(img,d,{mode,shape:'contour',dilateRadius:3,featherRadius:2,strength:1});
  const meta=await sharp(r.buffer).metadata();
  const px=await sharp(r.buffer).extract({left:200,top:200,width:1,height:1}).raw().toBuffer();
  console.log(mode,'->',meta.width+'x'+meta.height,'px@200,200=',[...px]);
}
})().catch(e=>{console.error('FAIL',e);process.exit(1)});
