import { pipeline, env, AutoModel, AutoProcessor, RawImage } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";
const $ = id => document.getElementById(id);
const state={source:null,sourceUrl:null,fileName:"rebag-photo",width:0,height:0,mask:null,baseMask:null,background:"transparent",backgroundImage:null,mode:"erase",brushSize:32,edgeCleanup:42,decontaminate:100,cleanedSource:null,remover:null,ben2Remover:null,currentModel:"ormbg",mattingModel:null,mattingProcessor:null,modelLoading:false,busy:false,pointerDown:false,lastPoint:null};
const preview=$("previewCanvas"),brush=$("brushCanvas"),previewCtx=preview.getContext("2d",{willReadFrequently:true}),brushCtx=brush.getContext("2d"),toastEl=$("toast");
function toast(message){toastEl.textContent=message;toastEl.classList.add("show");clearTimeout(toast._timer);toast._timer=setTimeout(()=>toastEl.classList.remove("show"),3200)}
function setBusy(busy,label){state.busy=busy;$("removeButton").disabled=busy||!state.source;$("ben2Button").disabled=busy||!state.source;$("removeButton").textContent=busy?(label||"Memproses…"):"✦ Hapus background otomatis"}
function setProgress(value,label){$("progressWrap").hidden=false;$("progressBar").style.width=Math.max(0,Math.min(100,value))+"%";$("progressText").textContent=label}
function finishProgress(){$("progressWrap").hidden=true;$("progressBar").style.width="0%"}
function imageFromFile(file){return new Promise((resolve,reject)=>{const url=URL.createObjectURL(file),img=new Image();img.onload=()=>resolve({img,url});img.onerror=()=>{URL.revokeObjectURL(url);reject(new Error("File foto tidak dapat dibuka."))};img.src=url})}
function createCanvas(width,height){const c=document.createElement("canvas");c.width=width;c.height=height;return c}
function buildSubjectCanvas(){const c=createCanvas(state.width,state.height),ctx=c.getContext("2d");ctx.drawImage(state.cleanedSource||state.source,0,0,state.width,state.height);if(state.mask){ctx.globalCompositeOperation="destination-in";ctx.drawImage(state.mask,0,0)}return c}
function drawBackground(ctx){if(state.background==="transparent")return;if(state.background instanceof HTMLImageElement){const bg=state.background,scale=Math.max(state.width/bg.naturalWidth,state.height/bg.naturalHeight),w=bg.naturalWidth*scale,h=bg.naturalHeight*scale;ctx.drawImage(bg,(state.width-w)/2,(state.height-h)/2,w,h)}else{ctx.fillStyle=state.background;ctx.fillRect(0,0,state.width,state.height)}}
function render(){if(!state.source)return;previewCtx.clearRect(0,0,preview.width,preview.height);drawBackground(previewCtx);previewCtx.drawImage(buildSubjectCanvas(),0,0)}
function showEditor(){if(!state.source)return;$("removeButton").disabled=state.busy;$("emptyState").hidden=true;preview.width=brush.width=state.width;preview.height=brush.height=state.height;preview.classList.add("visible");brush.classList.add("visible");brush.style.pointerEvents=state.mask?"auto":"none";brushCtx.clearRect(0,0,brush.width,brush.height);render();$("dimensions").textContent=state.width.toLocaleString("id-ID")+" × "+state.height.toLocaleString("id-ID")+" px";$("previewStatus").textContent=state.mask?"Background terhapus":"Foto asli";$("zoomLabel").textContent="Rasio dan resolusi asli";$("resetMask").disabled=!state.mask;enableExports()}
function enableExports(){["exportPng","exportJpg","exportPdf"].forEach(id=>$(id).disabled=!state.source||state.busy)}
async function applyEdgeCleanup(){
  if(!state.baseMask||!state.mask)return;
  const source=state.baseMask.getContext("2d",{willReadFrequently:true}).getImageData(0,0,state.width,state.height);
  const target=state.mask.getContext("2d"),pixels=source.data,strength=state.edgeCleanup/100;
  const width=state.width,height=state.height,rowStride=width*4;
  const originalAlpha=new Uint8ClampedArray(width*height);
  for(let p=0,j=3;p<originalAlpha.length;p++,j+=4)originalAlpha[p]=pixels[j];
  // BEN2 already returns a soft alpha matte; use gentler cleanup to protect fine strands.\n  const effectiveStrength=state.currentModel==="ben2"?strength*.25:strength;\n  const cutoff=0.012+effectiveStrength*0.075,exponent=1+effectiveStrength*.42;
  for(let y=0;y<height;y++){
    if((y&15)===0)await new Promise(resolve=>requestAnimationFrame(resolve));
    const rowStart=y*rowStride;
    for(let x=0,i=rowStart;x<width;x++,i+=4){
      const alpha=originalAlpha[y*width+x]/255;
      if(alpha<=0.008){pixels[i+3]=0;continue}
      if(effectiveStrength===0)continue;
      let normalized=Math.pow(Math.max(0,Math.min(1,(alpha-cutoff)/(1-cutoff))),exponent);
      // Spend neighborhood work only on translucent edge pixels.
      if(alpha<.22){
        let nearby=0;
        for(let dy=-2;dy<=2;dy++){
          const yy=y+dy;if(yy<0||yy>=height)continue;
          for(let dx=-2;dx<=2;dx++){
            const xx=x+dx;if(xx<0||xx>=width)continue;
            const value=originalAlpha[yy*width+xx]/255;
            if(value>nearby)nearby=value;
          }
        }
        if(alpha<.16&&nearby<.48)normalized*=Math.max(.12,nearby/.48);
        if(alpha<.045&&nearby<.2)normalized=0;
      }
      pixels[i+3]=Math.round(normalized*255);
    }
  }
  target.putImageData(source,0,0);
}
async function decontaminateEdgeColors(){
  if(!state.source||!state.mask)return;
  const width=state.width,height=state.height;
  const sourceCanvas=createCanvas(width,height),ctx=sourceCanvas.getContext("2d",{willReadFrequently:true});
  ctx.drawImage(state.source,0,0,width,height);
  const image=ctx.getImageData(0,0,width,height);
  const alpha=state.mask.getContext("2d",{willReadFrequently:true}).getImageData(0,0,width,height).data;
  const src=image.data,strength=state.decontaminate/100;
  if(strength<=0){state.cleanedSource=sourceCanvas;return}
  // Search nearby transparent pixels at several distances. A shorter first
  // radius is important for fine hair, where the old background is very local.
  const offsets=[[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[1,-1],[-1,1],[1,1]];
  const radii=[1,2,3,5,8,12,18],maxRadius=18;
  const samplesR=[],samplesG=[],samplesB=[];
  if(width<=maxRadius*2||height<=maxRadius*2){state.cleanedSource=sourceCanvas;return}
  const median=arr=>{arr.sort((a,b)=>a-b);return arr[arr.length>>1]};
  for(let y=maxRadius;y<height-maxRadius;y++){
    if((y&15)===0)await new Promise(resolve=>requestAnimationFrame(resolve));
    let i=(y*width+maxRadius)*4;
    for(let x=maxRadius;x<width-maxRadius;x++,i+=4){
      const a=alpha[i+3]/255;
      // Include the faintest retained strands, but leave fully opaque interior alone.
      if(a<.012||a>.992)continue;
      let found=false;
      for(const radius of radii){
        samplesR.length=samplesG.length=samplesB.length=0;
        for(let k=0;k<8;k++){
          const [dx,dy]=offsets[k],ni=((y+dy*radius)*width+(x+dx*radius))*4;
          if(alpha[ni+3]>5)continue;
          samplesR.push(src[ni]);samplesG.push(src[ni+1]);samplesB.push(src[ni+2]);
        }
        if(samplesR.length>=2){found=true;break}
      }
      if(!found)continue;
      const br=median(samplesR),bg=median(samplesG),bb=median(samplesB);
      // Unmix the original foreground/background blend. A lower denominator
      // removes more residual spill on faint strands; the correction floor
      // below limits noise amplification when alpha is extremely small.
      const denom=Math.max(.035,a);
      const fr=Math.max(0,Math.min(255,(src[i]-(1-a)*br)/denom));
      const fg=Math.max(0,Math.min(255,(src[i+1]-(1-a)*bg)/denom));
      const fb=Math.max(0,Math.min(255,(src[i+2]-(1-a)*bb)/denom));
      const confidence=Math.min(1,samplesR.length/5);
      const lowAlphaProtection=Math.min(1,Math.max(.58,a/.08));
      const correction=Math.min(1,strength*confidence*lowAlphaProtection);
      src[i]=Math.round(src[i]+(fr-src[i])*correction);
      src[i+1]=Math.round(src[i+1]+(fg-src[i+1])*correction);
      src[i+2]=Math.round(src[i+2]+(fb-src[i+2])*correction);
    }
  }
  ctx.putImageData(image,0,0);
  state.cleanedSource=sourceCanvas;
}
async function refineHairMatting({automatic=false}={}){
  if(!state.source||!state.mask||state.busy)return;
  const button=$("hairMattingButton");
  setBusy(true,"Meningkatkan detail rambut…");button.disabled=true;
  let imageUrl=null;
  try{
    setProgress(12,"Memuat model portrait matting…");
    if(!state.mattingModel||!state.mattingProcessor){
      [state.mattingModel,state.mattingProcessor]=await Promise.all([
        AutoModel.from_pretrained("Xenova/modnet",{dtype:"fp32"}),
        AutoProcessor.from_pretrained("Xenova/modnet")
      ]);
    }
    setProgress(42,"Menganalisis helaian rambut dan tepi subjek…");
    const scale=Math.min(1,1024/Math.max(state.width,state.height));
    const w=Math.max(1,Math.round(state.width*scale)),h=Math.max(1,Math.round(state.height*scale));
    const input=createCanvas(w,h);
    input.getContext("2d").drawImage(state.source,0,0,w,h);
    const blob=await new Promise((resolve,reject)=>input.toBlob(v=>v?resolve(v):reject(new Error("Gagal menyiapkan foto untuk matting.")),"image/png"));
    imageUrl=URL.createObjectURL(blob);
    const image=await RawImage.fromURL(imageUrl);
    const {pixel_values}=await state.mattingProcessor(image);
    const {output}=await state.mattingModel({input:pixel_values});
    if(!output||!output[0])throw new Error("Model tidak menghasilkan alpha matte.");
    const matte=await RawImage.fromTensor(output[0].mul(255).to("uint8")).resize(image.width,image.height);
    if(!matte.data||!matte.width||!matte.height||!matte.channels)throw new Error("Alpha matte tidak valid.");
    setProgress(78,"Menggabungkan matte rambut dengan deteksi subjek…");
    const refined=createCanvas(state.width,state.height),ctx=refined.getContext("2d");
    const pixels=ctx.createImageData(state.width,state.height),base=state.mask.getContext("2d",{willReadFrequently:true}).getImageData(0,0,state.width,state.height).data,out=pixels.data;
    // Fuse both masks: keep confident ORMBG foreground/background decisions,
    // and let MODNet refine only the uncertain silhouette and fine hair.
    const channels=matte.channels;
    for(let y=0;y<state.height;y++){
      if((y&7)===0)await new Promise(resolve=>requestAnimationFrame(resolve));
      const my=Math.min(matte.height-1,Math.floor(y*matte.height/state.height));
      for(let x=0,i=(y*state.width)*4;x<state.width;x++,i+=4){
        const mx=Math.min(matte.width-1,Math.floor(x*matte.width/state.width));
        const mi=(my*matte.width+mx)*channels;
        let value=matte.data[mi];
        if(channels>=3)value=(matte.data[mi]+matte.data[mi+1]+matte.data[mi+2])/3;
        const modnetAlpha=Math.max(0,Math.min(255,value));
        const baseAlpha=base[i+3];
        let fusedAlpha;
        if(baseAlpha<=6){
          // Reject weak background predictions while retaining confident fine hair.
          fusedAlpha=modnetAlpha>=35?Math.round(modnetAlpha*.72):0;
        }else{
          // Let MODNet soften hard segmentation edges as well as recover hair strands.
          fusedAlpha=Math.round(baseAlpha*.28+modnetAlpha*.72);
        }
        out[i]=out[i+1]=out[i+2]=255;
        out[i+3]=Math.max(0,Math.min(255,fusedAlpha));
      }
    }
    ctx.putImageData(pixels,0,0);
    state.baseMask=refined;state.mask=createCanvas(state.width,state.height);
    state.mask.getContext("2d").drawImage(refined,0,0);
    await applyEdgeCleanup();await decontaminateEdgeColors();render();
    $("previewStatus").textContent="Matte rambut dan tepi diperhalus";
    toast(automatic?"Penghapusan background dan perapian matte otomatis selesai.":"Perapian matte selesai. Periksa detail sebelum ekspor.");
  }catch(error){
    console.error(error);
    $("previewStatus").textContent="Background terhapus · matting tambahan dilewati";
    toast(automatic?"Background utama tetap tersedia. Perapian matte otomatis dilewati.":"Matting rambut tidak berhasil: "+(error.message||"coba lagi."));
  }finally{
    if(imageUrl)URL.revokeObjectURL(imageUrl);
    finishProgress();setBusy(false);button.disabled=!state.mask;enableExports();
  }
}
async function loadModel(){
  if(state.remover)return state.remover;
  if(state.modelLoading)throw new Error("Model AI sedang disiapkan. Tunggu sebentar.");
  state.modelLoading=true;
  try{
    setProgress(8,"Menyiapkan mesin AI…");
    env.allowLocalModels=false;
    env.useBrowserCache=true;
    let remover=null;
    if(navigator.gpu&&typeof navigator.gpu.requestAdapter==="function"){
      try{
        setProgress(18,"Memeriksa dukungan GPU…");
        const adapter=await navigator.gpu.requestAdapter();
        if(adapter&&adapter.features&&adapter.features.has("shader-f16")){
          setProgress(28,"Menyiapkan AI dengan GPU FP16…");
          remover=await pipeline("background-removal","Orangebot/ormbg-ONNX",{
            device:"webgpu",
            dtype:"fp16"
          });
        }else{
          console.info("WebGPU adapter/FP16 tidak tersedia; langsung menggunakan CPU/WASM.");
        }
      }catch(error){
        console.warn("WebGPU tidak dapat digunakan; beralih ke CPU/WASM.",error);
      }
    }else{
      console.info("WebGPU tidak didukung browser; menggunakan CPU/WASM.");
    }
    if(!remover){
      setProgress(32,"Menyiapkan AI mode kompatibel (CPU/WASM)…");
      remover=await pipeline("background-removal","Orangebot/ormbg-ONNX",{
        device:"wasm",
        dtype:"uint8"
      });
    }
    state.remover=remover;
    return remover;
  }finally{
    state.modelLoading=false;
  }
}
async function loadBen2Model(){
  if(state.ben2Remover)return state.ben2Remover;
  if(state.modelLoading)throw new Error("Model AI sedang disiapkan. Tunggu sebentar.");
  state.modelLoading=true;
  try{
    setProgress(10,"Menyiapkan model BEN2…");
    env.allowLocalModels=false;
    env.useBrowserCache=true;
    let remover=null;
    if(navigator.gpu&&typeof navigator.gpu.requestAdapter==="function"){
      try{
        const adapter=await navigator.gpu.requestAdapter();
        if(adapter&&adapter.features&&adapter.features.has("shader-f16")){
          setProgress(24,"Memuat BEN2 untuk GPU…");
          remover=await pipeline("background-removal","onnx-community/BEN2-ONNX",{device:"webgpu",dtype:"fp16"});
        }
      }catch(error){console.warn("BEN2 WebGPU gagal; mencoba WASM.",error)}
    }
    if(!remover){
      setProgress(28,"Memuat BEN2 mode kompatibel (WASM)…");
      remover=await pipeline("background-removal","onnx-community/BEN2-ONNX",{device:"wasm",dtype:"q8"});
    }
    state.ben2Remover=remover;
    return remover;
  }finally{state.modelLoading=false}
}
async function removeBackground({autoRefine=false,model="ormbg"}={}){
  if(!state.source||state.busy)return;
  setBusy(true,model==="ben2"?"Menyiapkan BEN2…":"Menyiapkan AI…");
  try{
    const remover=model==="ben2"?await loadBen2Model():await loadModel();
    setProgress(55,model==="ben2"?"BEN2 menganalisis rambut dan tepian…":"AI sedang mengenali objek…");
    const maxSide=1024,scale=Math.min(1,maxSide/Math.max(state.width,state.height));
    const inferenceWidth=Math.max(1,Math.round(state.width*scale)),inferenceHeight=Math.max(1,Math.round(state.height*scale));
    const inferenceCanvas=createCanvas(inferenceWidth,inferenceHeight);
    inferenceCanvas.getContext("2d").drawImage(state.source,0,0,inferenceWidth,inferenceHeight);
    const inferenceBlob=await new Promise((resolve,reject)=>inferenceCanvas.toBlob(blob=>blob?resolve(blob):reject(new Error("Gagal menyiapkan gambar untuk AI.")),"image/png"));
    const result=model==="ben2"?await remover([inferenceBlob]):await remover(inferenceBlob);
    const cutout=Array.isArray(result)?result[0]:result;
    if(!cutout)throw new Error("AI tidak menghasilkan mask. Coba foto lain.");
    let cutoutCanvas;
    if(typeof cutout.toCanvas==="function")cutoutCanvas=cutout.toCanvas();
    else if(cutout instanceof HTMLCanvasElement)cutoutCanvas=cutout;
    else throw new Error("Format hasil AI tidak dikenali.");
    const cutoutCtx=cutoutCanvas.getContext("2d",{willReadFrequently:true});
    const pixels=cutoutCtx.getImageData(0,0,cutoutCanvas.width,cutoutCanvas.height),data=pixels.data;
    let hasAlpha=false;
    for(let i=3;i<data.length;i+=4){if(data[i]<250){hasAlpha=true;break}}
    if(!hasAlpha){
      for(let i=0;i<data.length;i+=4)data[i+3]=Math.round(data[i]*.299+data[i+1]*.587+data[i+2]*.114);
      cutoutCtx.putImageData(pixels,0,0);
    }
    setProgress(88,"Menyusun matte pada resolusi asli…");
    const mask=createCanvas(state.width,state.height);
    mask.getContext("2d").drawImage(cutoutCanvas,0,0,state.width,state.height);
    state.currentModel=model;
    state.cleanedSource=null;
    state.mask=mask;
    state.baseMask=createCanvas(state.width,state.height);
    state.baseMask.getContext("2d").drawImage(mask,0,0);
    $("hairMattingButton").disabled=false;
    await applyEdgeCleanup();
    await decontaminateEdgeColors();
    brush.style.pointerEvents="auto";
    render();
    $("previewStatus").textContent=model==="ben2"?"BEN2 · matte rambut detail":"ORMBG · background terhapus";
    $("resetMask").disabled=false;
    finishProgress();
    toast(model==="ben2"?"BEN2 selesai. Periksa rambut dan tepian; model alternatif ini masih tahap uji.":"Background berhasil diproses. Periksa tepi objek sebelum ekspor.");
  }catch(error){
    console.error(error);
    finishProgress();
    toast((model==="ben2"?"BEN2 gagal: ":"")+(error.message||"Gagal memproses foto. Periksa koneksi dan coba lagi."));
  }finally{
    setBusy(false);
    enableExports();
    if(autoRefine&&state.mask)await refineHairMatting({automatic:true});
  }
}function setBackground(value){state.background=value;document.querySelectorAll(".swatch").forEach(b=>b.classList.toggle("selected",b.dataset.bg===value));render()}
let pickerHue=210,pickerSaturation=100,pickerValue=100;
function hsvToHex(h,s,v){s/=100;v/=100;const c=v*s,x=c*(1-Math.abs((h/60)%2-1)),m=v-c;let r=0,g=0,b=0;if(h<60){r=c;g=x}else if(h<120){r=x;g=c}else if(h<180){g=c;b=x}else if(h<240){g=x;b=c}else if(h<300){r=x;b=c}else{r=c;b=x}return "#"+[r,g,b].map(n=>Math.round((n+m)*255).toString(16).padStart(2,"0")).join("")}
function applyLivePicker(){const color=hsvToHex(pickerHue,pickerSaturation,pickerValue);$("saturationPicker").style.backgroundColor="hsl("+pickerHue+",100%,50%)";$("colorPickerHandle").style.left=pickerSaturation+"%";$("colorPickerHandle").style.top=(100-pickerValue)+"%";$("customColorPreview").style.background=color;setBackground(color)}
function updatePickerFromPointer(event,element,kind){const rect=element.getBoundingClientRect();if(kind==="hue"){pickerHue=Math.max(0,Math.min(359,((event.clientX-rect.left)/rect.width)*360));}else{pickerSaturation=Math.max(0,Math.min(100,((event.clientX-rect.left)/rect.width)*100));pickerValue=100-Math.max(0,Math.min(100,((event.clientY-rect.top)/rect.height)*100));}applyLivePicker()}
function bindLivePicker(element,kind){element.addEventListener("pointerenter",e=>updatePickerFromPointer(e,element,kind));element.addEventListener("pointermove",e=>updatePickerFromPointer(e,element,kind));element.addEventListener("pointerdown",e=>{element.setPointerCapture(e.pointerId);updatePickerFromPointer(e,element,kind)});element.addEventListener("pointerup",e=>{if(element.hasPointerCapture(e.pointerId))element.releasePointerCapture(e.pointerId)})}

function downloadBlob(blob,name){if(!(blob instanceof Blob)||blob.size===0)throw new Error("File PNG kosong atau gagal dibuat.");const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=name;a.style.display="none";a.rel="noopener";document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),60000)}
function canvasBlob(canvas,type,quality){return new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(new Error("Gagal membuat file ekspor.")),type,quality))}
function makeExportCanvas(forJpg){const c=createCanvas(state.width,state.height),ctx=c.getContext("2d");if(forJpg&&state.background==="transparent"){ctx.fillStyle="#fff";ctx.fillRect(0,0,state.width,state.height)}else drawBackground(ctx);ctx.drawImage(buildSubjectCanvas(),0,0);return c}
async function exportPng(){if(!state.source||state.busy)return;try{downloadBlob(await canvasBlob(makeExportCanvas(false),"image/png"),state.fileName+"-rebag.png");toast("PNG berhasil dibuat pada resolusi asli.")}catch(e){toast(e.message)}}
async function exportJpg(){if(!state.source||state.busy)return;try{downloadBlob(await canvasBlob(makeExportCanvas(true),"image/jpeg",.95),state.fileName+"-rebag.jpg");toast("JPG kualitas 95% berhasil dibuat.")}catch(e){toast(e.message)}}
async function exportPdf(){if(!state.source||state.busy)return;const button=$("exportPdf");button.disabled=true;button.textContent="Menyiapkan…";try{const {jsPDF}=await import("https://cdn.jsdelivr.net/npm/jspdf@2.5.2/+esm"),canvas=makeExportCanvas(false),data=canvas.toDataURL("image/png"),dpi=300,pageWidth=state.width*72/dpi,pageHeight=state.height*72/dpi,orientation=pageWidth>pageHeight?"landscape":"portrait";const pdf=new jsPDF({orientation,unit:"pt",format:[pageWidth,pageHeight],compress:true});pdf.addImage(data,"PNG",0,0,pageWidth,pageHeight,undefined,"FAST");pdf.save(state.fileName+"-rebag.pdf");toast("PDF satu foto satu halaman berhasil dibuat.")}catch(e){console.error(e);toast("PDF gagal dibuat: "+(e.message||"coba lagi."))}finally{button.textContent="Unduh";enableExports()}}
async function resetMask(){if(!state.baseMask)return;state.mask=createCanvas(state.width,state.height);state.mask.getContext("2d").drawImage(state.baseMask,0,0);await applyEdgeCleanup();await decontaminateEdgeColors();render();toast("Mask AI dipulihkan dengan koreksi tepi saat ini.")}
function getCanvasPoint(event){const r=brush.getBoundingClientRect();return{x:(event.clientX-r.left)*state.width/r.width,y:(event.clientY-r.top)*state.height/r.height}}
function paintStroke(point){if(!state.mask||!state.baseMask)return;const ctx=state.mask.getContext("2d"),scale=state.width/brush.getBoundingClientRect().width,radius=state.brushSize*scale/2;ctx.save();ctx.lineCap="round";ctx.lineJoin="round";ctx.lineWidth=radius*2;if(state.mode==="erase"){ctx.globalCompositeOperation="destination-out";ctx.strokeStyle="rgba(0,0,0,1)";ctx.beginPath();if(state.lastPoint){ctx.moveTo(state.lastPoint.x,state.lastPoint.y);ctx.lineTo(point.x,point.y)}else{ctx.moveTo(point.x,point.y);ctx.lineTo(point.x+.01,point.y)}ctx.stroke()}else{const path=new Path2D();const from=state.lastPoint||point,dx=point.x-from.x,dy=point.y-from.y,dist=Math.hypot(dx,dy),steps=Math.max(1,Math.ceil(dist/Math.max(1,radius*.45)));for(let i=0;i<=steps;i++){const x=from.x+dx*i/steps,y=from.y+dy*i/steps;path.moveTo(x+radius,y);path.arc(x,y,radius,0,Math.PI*2)}ctx.beginPath();ctx.clip(path);ctx.globalCompositeOperation="source-over";ctx.drawImage(state.baseMask,0,0)}ctx.restore()}
function onBrushDown(e){if(!state.mask)return;state.pointerDown=true;brush.setPointerCapture(e.pointerId);state.lastPoint=null;const p=getCanvasPoint(e);paintStroke(p);state.lastPoint=p;render()}
function onBrushMove(e){if(!state.pointerDown)return;const p=getCanvasPoint(e);paintStroke(p);state.lastPoint=p;render()}
async function onBrushUp(){state.pointerDown=false;state.lastPoint=null;if(state.mask){await decontaminateEdgeColors();render()}}
async function loadMainPhoto(file){if(!file||!/^image\/(jpeg|png|webp)$/.test(file.type)){toast("Pilih foto JPG, PNG, atau WebP.");return}try{const loaded=await imageFromFile(file);if(state.sourceUrl)URL.revokeObjectURL(state.sourceUrl);state.source=loaded.img;state.sourceUrl=loaded.url;state.width=loaded.img.naturalWidth;state.height=loaded.img.naturalHeight;state.fileName=(file.name||"rebag-photo").replace(/\.[^.]+$/,"").replace(/[^\w-]+/g,"-");state.mask=null;state.baseMask=null;state.cleanedSource=null;$("hairMattingButton").disabled=true;$("fileMeta").textContent=file.name+" • "+state.width.toLocaleString("id-ID")+" × "+state.height.toLocaleString("id-ID")+" px";$("previewStatus").textContent="Foto siap diedit";showEditor();toast("Foto diunggah. AI menghapus background; perapian rambut lanjutan tersedia lewat tombol manual.");await removeBackground()}catch(e){toast(e.message)}}
$("photoInput").addEventListener("change",e=>loadMainPhoto(e.target.files[0]));
$("dropzone").addEventListener("dragover",e=>{e.preventDefault();$("dropzone").classList.add("dragover")});
$("dropzone").addEventListener("dragleave",()=>$("dropzone").classList.remove("dragover"));
$("dropzone").addEventListener("drop",e=>{e.preventDefault();$("dropzone").classList.remove("dragover");loadMainPhoto(e.dataTransfer.files[0])});
$("removeButton").addEventListener("click",()=>removeBackground({model:"ormbg"}));\n$("ben2Button").addEventListener("click",()=>removeBackground({model:"ben2"}));
document.querySelectorAll(".swatch").forEach(b=>{b.addEventListener("pointerenter",()=>setBackground(b.dataset.bg));b.addEventListener("click",()=>setBackground(b.dataset.bg))});
bindLivePicker($("saturationPicker"),"saturation");bindLivePicker($("huePicker"),"hue");applyLivePicker();
$("backgroundInput").addEventListener("change",async e=>{const file=e.target.files[0];if(!file)return;try{const loaded=await imageFromFile(file);state.backgroundImage=loaded.img;$("backgroundImageMeta").textContent="Background: "+file.name;setBackground(loaded.img);toast("Background gambar berhasil diterapkan.")}catch(err){toast(err.message)}});
$("eraseMode").addEventListener("click",()=>{state.mode="erase";$("eraseMode").classList.add("active");$("restoreMode").classList.remove("active")});
$("restoreMode").addEventListener("click",()=>{state.mode="restore";$("restoreMode").classList.add("active");$("eraseMode").classList.remove("active")});
$("brushSize").addEventListener("input",e=>{state.brushSize=Number(e.target.value);$("brushValue").textContent=state.brushSize+" px"});let edgeTuneTimer=null;function scheduleEdgeTune(){clearTimeout(edgeTuneTimer);edgeTuneTimer=setTimeout(async()=>{if(!state.baseMask)return;state.mask=createCanvas(state.width,state.height);state.mask.getContext("2d").drawImage(state.baseMask,0,0);await applyEdgeCleanup();await decontaminateEdgeColors();render()},180)}$("edgeCleanup").addEventListener("input",e=>{state.edgeCleanup=Number(e.target.value);$("edgeCleanupValue").textContent=state.edgeCleanup+"%";scheduleEdgeTune()});$("decontaminate").addEventListener("input",e=>{state.decontaminate=Number(e.target.value);$("decontaminateValue").textContent=state.decontaminate+"%";scheduleEdgeTune()});
$("hairMattingButton").addEventListener("click",refineHairMatting);$("resetMask").addEventListener("click",resetMask);
brush.addEventListener("pointerdown",onBrushDown);brush.addEventListener("pointermove",onBrushMove);brush.addEventListener("pointerup",onBrushUp);brush.addEventListener("pointercancel",onBrushUp);
$("exportPng").addEventListener("click",exportPng);$("exportJpg").addEventListener("click",exportJpg);$("exportPdf").addEventListener("click",exportPdf);
$("fitButton").addEventListener("click",()=>{if(state.source){showEditor();toast("Pratinjau disesuaikan ke area.")}});
$("resetView").addEventListener("click",()=>{if(state.source)render()});
// Biarkan browser menampilkan prompt instalasi PWA secara native.
if("serviceWorker"in navigator)window.addEventListener("load",()=>navigator.serviceWorker.register("./sw.js").catch(e=>console.warn("Service worker gagal didaftarkan:",e)));
enableExports();