self.onmessage=event=>{
  const {id,type,width,height,strength,sourceBuffer,alphaBuffer}=event.data||{};
  if(type!=="decontaminate")return;
  try{
    const src=new Uint8ClampedArray(sourceBuffer);
    const alpha=new Uint8ClampedArray(alphaBuffer);
    const directions=[[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1]];
    const radii=[1,2,3,4,6,8,12,16,22,30];
    const samplesR=[],samplesG=[],samplesB=[];
    const median=values=>{
      values.sort((a,b)=>a-b);
      return values[values.length>>1];
    };
    for(let y=0;y<height;y++){
      const row=y*width*4;
      for(let x=0,i=row;x<width;x++,i+=4){
        const a=alpha[i+3]/255;
        if(a<.012||a>.992)continue;
        samplesR.length=samplesG.length=samplesB.length=0;
        for(const [dx,dy] of directions){
          for(const radius of radii){
            const sx=x+dx*radius,sy=y+dy*radius;
            if(sx<0||sx>=width||sy<0||sy>=height)break;
            const si=(sy*width+sx)*4;
            if(alpha[si+3]<=8){
              samplesR.push(src[si]);
              samplesG.push(src[si+1]);
              samplesB.push(src[si+2]);
              break;
            }
          }
        }
        if(samplesR.length<3)continue;
        const br=median(samplesR),bg=median(samplesG),bb=median(samplesB);
        const denom=Math.max(.045,a);
        const fr=Math.max(0,Math.min(255,(src[i]-(1-a)*br)/denom));
        const fg=Math.max(0,Math.min(255,(src[i+1]-(1-a)*bg)/denom));
        const fb=Math.max(0,Math.min(255,(src[i+2]-(1-a)*bb)/denom));
        const confidence=Math.min(1,samplesR.length/6);
        const alphaSafety=Math.min(1,Math.max(.42,a/.10));
        const correction=Math.min(.92,strength*confidence*alphaSafety);
        src[i]=Math.round(src[i]+(fr-src[i])*correction);
        src[i+1]=Math.round(src[i+1]+(fg-src[i+1])*correction);
        src[i+2]=Math.round(src[i+2]+(fb-src[i+2])*correction);
      }
    }
    self.postMessage({id,buffer:src.buffer},[src.buffer]);
  }catch(error){
    self.postMessage({id,error:error?.message||"Edge decontamination failed."});
  }
};