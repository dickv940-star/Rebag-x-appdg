import { pipeline, env, AutoModel, AutoProcessor, RawImage } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";
const $ = id => document.getElementById(id);
const state={source:null,sourceUrl:null,fileName:"rebag-photo",width:0,height:0,mask:null,baseMask:null,background:"transparent",backgroundImage:null,mode:"erase",brushSize:32,edgeCleanup:18,decontaminate:100,cleanedSource:null,remover:null,mattingModel:null,mattingProcessor:null,modelLoading:false,busy:false,pointerDown:false,lastPoint:null};
const preview=$("previewCanvas"),brush=$("brushCanvas"),previewCtx=preview.getContext("2d",{willReadFrequently:true}),brushCtx=brush.getContext("2d"),toastEl=$("toast");
function toast(message){toastEl.textContent=message;toastEl.classList.add("show");clearTimeout(toast._timer);toast._timer=setTimeout(()=>toastEl.classList.remove("show"),3200)}
function setBusy(busy,label){state.busy=busy;$("removeButton").disabled=busy||!state.source;$("removeButton").textContent=busy?(label||"Memproses…"):"✦ Hapus background otomatis"}
function setProgress(value,label){$("progressWrap").hidden=false;$("progressBar").style.width=Math.max(0,Math.min(100,value))+"%";$("progressText").textContent=label}
function finishProgress(){$("progressWrap").hidden=true;$("progressBar").style.width="0%"}
function imageFromFile(file){return new Promise((resolve,reject)=>{const url=URL.createObjectURL(file),img=new Image();img.onload=()=>resolve({img,url});img.onerror=()=>{URL.revokeObjectURL(url);reject(new Error("File foto tidak dapat dibuka."))};img.src=url})}
function createCanvas(width,height){const c=document.createElement("canvas");c.width=width;c.height=height;return c}
function buildSubjectCanvas(){const c=createCanvas(state.width,state.height),ctx=c.getContext("2d");ctx.drawImage(state.cleanedSource||state.source,0,0,state.width,state.height);if(state.mask){ctx.globalCompositeOperation="destination-in";ctx.drawImage(state.mask,0,0)}return c}
function drawBackground(ctx){if(state.background==="transparent")return;if(state.background instanceof HTMLImageElement){const bg=state.background,scale=Math.max(state.width/bg.naturalWidth,state.height/bg.naturalHeight),w=bg.naturalWidth*scale,h=bg.naturalHeight*scale;ctx.drawImage(bg,(state.width-w)/2,(state.height-h)/2,w,h)}else{ctx.fillStyle=state.background;ctx.fillRect(0,0,state.width,state.height)}}
function render(){if(!state.source)return;previewCtx.clearRect(0,0,preview.width,preview.height);drawBackground(previewCtx);previewCtx.drawImage(buildSubjectCanvas(),0,0)}
function showEditor(){if(!state.source)return;$("removeButton").disabled=state.busy;$("emptyState").hidden=true;preview.width=brush.width=state.width;preview.height=brush.height=state.height;preview.classList.add("visible");brush.classList.add("visible");brush.style.pointerEvents=state.mask?"auto":"none";brushCtx.clearRect(0,0,brush.width,brush.height);render();$("dimensions").textContent=state.width.toLocaleString("id-ID")+" × "+state.height.toLocaleString("id-ID")+" px";$("previewStatus").textContent=state.mask?"Background terhapus":"Foto asli";$("zoomLabel").textContent="Rasio dan resolusi asli";$("resetMask").disabled=!state.mask;enableExports()}
function enableExports(){["exportPng","exportJpg","exportPdf"].forEach(id=>$(id).disabled=!state.source||state.busy)}
function applyEdgeCleanup(){
  if(!state.baseMask||!state.mask)return;
  const source=state.baseMask.getContext("2d",{willReadFrequently:true}).getImageData(0,0,state.width,state.height);
  const target=state.mask.getContext("2d"),pixels=source.data,strength=state.edgeCleanup/100;
  // Preserve low-opacity hair strands while reducing faint background haze.
  const cutoff=strength*.025,exponent=1+strength*.16;
  for(let i=3;i<pixels.length;i+=4){
    const alpha=pixels[i]/255;
    if(alpha<=0){pixels[i]=0;continue}
    if(strength===0)continue;
    const normalized=Math.max(0,Math.min(1,(alpha-cutoff)/(1-cutoff)));
    pixels[i]=Math.round(Math.pow(normalized,exponent)*255);
  }
  target.putImageData(source,0,0);
}
function decontaminateEdgeColors(){
  if(!state.source||!state.mask)return;
  const width=state.width,height=state.height;
  const sourceCanvas=createCanvas(width,height),ctx=sourceCanvas.getContext("2d",{willReadFrequently:true});
  ctx.drawImage(state.source,0,0,width,height);
  const image=ctx.getImageData(0,0,width,height);
  const mask=state.mask.getContext("2d",{willReadFrequently:true}).getImageData(0,0,width,height);
  const src=image.data,alpha=mask.data,strength=state.decontaminate/100;
  if(strength<=0){state.cleanedSource=sourceCanvas;return}
  // Estimate old background from nearby fully transparent pixels. Prefer the
  // nearest radius with enough samples to avoid pulling in distant colors.
  const rings=[2,4];
  for(let y=11;y<height-11;y++){
    let i=(y*width+11)*4;
    for(let x=11;x<width-11;x++,i+=4){
      const a=alpha[i+3]/255;
      if(a<0.07||a>0.94)continue;
      let bgR=0,bgG=0,bgB=0,total=0,samples=0;
      for(const radius of rings){
        bgR=bgG=bgB=total=samples=0;
        const directions=8;
        for(let step=0;step<directions;step++){
          const angle=(Math.PI*2*step)/directions;
          const dx=Math.round(Math.cos(angle)*radius),dy=Math.round(Math.sin(angle)*radius);
          const ni=i+(dy*width+dx)*4,na=alpha[ni+3]/255;
          if(na>0.02)continue;
          const weight=1/(radius*radius);
          bgR+=src[ni]*weight;bgG+=src[ni+1]*weight;bgB+=src[ni+2]*weight;
          total+=weight;samples++;
        }
        if(samples>=2)break;
      }
      if(samples<2||total===0)continue;
      const inv=1/total,bg=[bgR*inv,bgG*inv,bgB*inv],denom=Math.max(0.16,a);
      const foreground=[
        Math.max(0,Math.min(255,(src[i]-(1-a)*bg[0])/denom)),
        Math.max(0,Math.min(255,(src[i+1]-(1-a)*bg[1])/denom)),
        Math.max(0,Math.min(255,(src[i+2]-(1-a)*bg[2])/denom))
      ];
      const sampleConfidence=Math.min(1,samples/3);
      const alphaConfidence=Math.min(1,Math.max(.22,(a-.07)/.28));
      const correction=Math.min(1,strength*sampleConfidence*alphaConfidence);
      src[i]=Math.round(src[i]+(foreground[0]-src[i])*correction);
      src[i+1]=Math.round(src[i+1]+(foreground[1]-src[i+1])*correction);
      src[i+2]=Math.round(src[i+2]+(foreground[2]-src[i+2])*correction);
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
    const scale=Math.min(1,1280/Math.max(state.width,state.height));
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
    // Use MODNet's predicted alpha matte directly, not the RGB channels of a cutout.
    // MODNet is the dedicated portrait matting model; use its alpha directly for natural hair edges.
    const mix=1,channels=matte.channels;
    for(let y=0;y<state.height;y++){
      const my=Math.min(matte.height-1,Math.floor(y*matte.height/state.height));
      for(let x=0,i=(y*state.width)*4;x<state.width;x++,i+=4){
        const mx=Math.min(matte.width-1,Math.floor(x*matte.width/state.width));
        const mi=(my*matte.width+mx)*channels;
        let value=matte.data[mi];
        if(channels>=3)value=(matte.data[mi]+matte.data[mi+1]+matte.data[mi+2])/3;
        const modnetAlpha=Math.max(0,Math.min(255,value));
        out[i]=out[i+1]=out[i+2]=255;
        out[i+3]=Math.round(base[i+3]*(1-mix)+modnetAlpha*mix);
      }
    }
    ctx.putImageData(pixels,0,0);
    state.baseMask=refined;state.mask=createCanvas(state.width,state.height);
    state.mask.getContext("2d").drawImage(refined,0,0);
    applyEdgeCleanup();decontaminateEdgeColors();render();
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
async function removeBackground({autoRefine=false}={}){if(!state.source||state.busy)return;setBusy(true,"Menyiapkan AI…");try{const remover=await loadModel();setProgress(55,"AI sedang mengenali objek…");const maxSide=1024,scale=Math.min(1,maxSide/Math.max(state.width,state.height)),inferenceWidth=Math.max(1,Math.round(state.width*scale)),inferenceHeight=Math.max(1,Math.round(state.height*scale)),inferenceCanvas=createCanvas(inferenceWidth,inferenceHeight);inferenceCanvas.getContext("2d").drawImage(state.source,0,0,inferenceWidth,inferenceHeight);const inferenceBlob=await new Promise((resolve,reject)=>inferenceCanvas.toBlob(blob=>blob?resolve(blob):reject(new Error("Gagal menyiapkan gambar untuk AI.")),"image/png"));const result=await remover(inferenceBlob),cutout=Array.isArray(result)?result[0]:result;if(!cutout)throw new Error("AI tidak menghasilkan mask. Coba foto lain.");let cutoutCanvas;if(typeof cutout.toCanvas==="function")cutoutCanvas=cutout.toCanvas();else if(cutout instanceof HTMLCanvasElement)cutoutCanvas=cutout;else throw new Error("Format hasil AI tidak dikenali.");const cutoutCtx=cutoutCanvas.getContext("2d",{willReadFrequently:true}),pixels=cutoutCtx.getImageData(0,0,cutoutCanvas.width,cutoutCanvas.height),data=pixels.data;let hasAlpha=false;for(let i=3;i<data.length;i+=4){if(data[i]<250){hasAlpha=true;break}}if(!hasAlpha){for(let i=0;i<data.length;i+=4)data[i+3]=Math.round(data[i]*.299+data[i+1]*.587+data[i+2]*.114);cutoutCtx.putImageData(pixels,0,0)}setProgress(88,"Menyusun hasil pada resolusi asli…");const mask=createCanvas(state.width,state.height);mask.getContext("2d").drawImage(cutoutCanvas,0,0,state.width,state.height);state.mask=mask;state.baseMask=createCanvas(state.width,state.height);state.baseMask.getContext("2d").drawImage(mask,0,0);$("hairMattingButton").disabled=false;applyEdgeCleanup();decontaminateEdgeColors();brush.style.pointerEvents="auto";render();$("previewStatus").textContent="Background terhapus";$("resetMask").disabled=false;finishProgress();toast("Background berhasil diproses. Periksa tepi objek sebelum ekspor.")}catch(error){console.error(error);finishProgress();toast(error.message||"Gagal memproses foto. Periksa koneksi dan coba lagi.")}finally{setBusy(false);enableExports();if(autoRefine&&state.mask)await refineHairMatting({automatic:true})}}
function setBackground(value){state.background=value;document.querySelectorAll(".swatch").forEach(b=>b.classList.toggle("selected",b.dataset.bg===value));render()}
let pickerHue=210,pickerSaturation=100,pickerValue=100;
function hsvToHex(h,s,v){s/=100;v/=100;const c=v*s,x=c*(1-Math.abs((h/60)%2-1)),m=v-c;let r=0,g=0,b=0;if(h<60){r=c;g=x}else if(h<120){r=x;g=c}else if(h<180){g=c;b=x}else if(h<240){g=x;b=c}else if(h<300){r=x;b=c}else{r=c;b=x}return "#"+[r,g,b].map(n=>Math.round((n+m)*255).toString(16).padStart(2,"0")).join("")}
function applyLivePicker(){const color=hsvToHex(pickerHue,pickerSaturation,pickerValue);$("saturationPicker").style.backgroundColor="hsl("+pickerHue+",100%,50%)";$("colorPickerHandle").style.left=pickerSaturation+"%";$("colorPickerHandle").style.top=(100-pickerValue)+"%";$("customColorPreview").style.background=color;setBackground(color)}
function updatePickerFromPointer(event,element,kind){const rect=element.getBoundingClientRect();if(kind==="hue"){pickerHue=Math.max(0,Math.min(359,((event.clientX-rect.left)/rect.width)*360));}else{pickerSaturation=Math.max(0,Math.min(100,((event.clientX-rect.left)/rect.width)*100));pickerValue=100-Math.max(0,Math.min(100,((event.clientY-rect.top)/rect.height)*100));}applyLivePicker()}
function bindLivePicker(element,kind){element.addEventListener("pointerenter",e=>updatePickerFromPointer(e,element,kind));element.addEventListener("pointermove",e=>updatePickerFromPointer(e,element,kind));element.addEventListener("pointerdown",e=>{element.setPointerCapture(e.pointerId);updatePickerFromPointer(e,element,kind)});element.addEventListener("pointerup",e=>{if(element.hasPointerCapture(e.pointerId))element.releasePointerCapture(e.pointerId)})}

function downloadBlob(blob,name){const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),2000)}
function canvasBlob(canvas,type,quality){return new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(new Error("Gagal membuat file ekspor.")),type,quality))}
function makeExportCanvas(forJpg){const c=createCanvas(state.width,state.height),ctx=c.getContext("2d");if(forJpg&&state.background==="transparent"){ctx.fillStyle="#fff";ctx.fillRect(0,0,state.width,state.height)}else drawBackground(ctx);ctx.drawImage(buildSubjectCanvas(),0,0);return c}
async function exportPng(){if(!state.source||state.busy)return;try{downloadBlob(await canvasBlob(makeExportCanvas(false),"image/png"),state.fileName+"-rebag.png");toast("PNG berhasil dibuat pada resolusi asli.")}catch(e){toast(e.message)}}
async function exportJpg(){if(!state.source||state.busy)return;try{downloadBlob(await canvasBlob(makeExportCanvas(true),"image/jpeg",.95),state.fileName+"-rebag.jpg");toast("JPG kualitas 95% berhasil dibuat.")}catch(e){toast(e.message)}}
async function exportPdf(){if(!state.source||state.busy)return;const button=$("exportPdf");button.disabled=true;button.textContent="Menyiapkan…";try{const {jsPDF}=await import("https://cdn.jsdelivr.net/npm/jspdf@2.5.2/+esm"),canvas=makeExportCanvas(false),data=canvas.toDataURL("image/png"),dpi=300,pageWidth=state.width*72/dpi,pageHeight=state.height*72/dpi,orientation=pageWidth>pageHeight?"landscape":"portrait";const pdf=new jsPDF({orientation,unit:"pt",format:[pageWidth,pageHeight],compress:true});pdf.addImage(data,"PNG",0,0,pageWidth,pageHeight,undefined,"FAST");pdf.save(state.fileName+"-rebag.pdf");toast("PDF satu foto satu halaman berhasil dibuat.")}catch(e){console.error(e);toast("PDF gagal dibuat: "+(e.message||"coba lagi."))}finally{button.textContent="Unduh";enableExports()}}
function resetMask(){if(!state.baseMask)return;state.mask=createCanvas(state.width,state.height);state.mask.getContext("2d").drawImage(state.baseMask,0,0);applyEdgeCleanup();decontaminateEdgeColors();render();toast("Mask AI dipulihkan dengan koreksi tepi saat ini.")}
function getCanvasPoint(event){const r=brush.getBoundingClientRect();return{x:(event.clientX-r.left)*state.width/r.width,y:(event.clientY-r.top)*state.height/r.height}}
function paintStroke(point){if(!state.mask||!state.baseMask)return;const ctx=state.mask.getContext("2d"),scale=state.width/brush.getBoundingClientRect().width,radius=state.brushSize*scale/2;ctx.save();ctx.lineCap="round";ctx.lineJoin="round";ctx.lineWidth=radius*2;if(state.mode==="erase"){ctx.globalCompositeOperation="destination-out";ctx.strokeStyle="rgba(0,0,0,1)";ctx.beginPath();if(state.lastPoint){ctx.moveTo(state.lastPoint.x,state.lastPoint.y);ctx.lineTo(point.x,point.y)}else{ctx.moveTo(point.x,point.y);ctx.lineTo(point.x+.01,point.y)}ctx.stroke()}else{const path=new Path2D();const from=state.lastPoint||point,dx=point.x-from.x,dy=point.y-from.y,dist=Math.hypot(dx,dy),steps=Math.max(1,Math.ceil(dist/Math.max(1,radius*.45)));for(let i=0;i<=steps;i++){const x=from.x+dx*i/steps,y=from.y+dy*i/steps;path.moveTo(x+radius,y);path.arc(x,y,radius,0,Math.PI*2)}ctx.beginPath();ctx.clip(path);ctx.globalCompositeOperation="source-over";ctx.drawImage(state.baseMask,0,0)}ctx.restore()}
function onBrushDown(e){if(!state.mask)return;state.pointerDown=true;brush.setPointerCapture(e.pointerId);state.lastPoint=null;const p=getCanvasPoint(e);paintStroke(p);state.lastPoint=p;render()}
function onBrushMove(e){if(!state.pointerDown)return;const p=getCanvasPoint(e);paintStroke(p);state.lastPoint=p;render()}
function onBrushUp(){state.pointerDown=false;state.lastPoint=null;if(state.mask){decontaminateEdgeColors();render()}}
async function loadMainPhoto(file){if(!file||!/^image\/(jpeg|png|webp)$/.test(file.type)){toast("Pilih foto JPG, PNG, atau WebP.");return}try{const loaded=await imageFromFile(file);if(state.sourceUrl)URL.revokeObjectURL(state.sourceUrl);state.source=loaded.img;state.sourceUrl=loaded.url;state.width=loaded.img.naturalWidth;state.height=loaded.img.naturalHeight;state.fileName=(file.name||"rebag-photo").replace(/\.[^.]+$/,"").replace(/[^\w-]+/g,"-");state.mask=null;state.baseMask=null;state.cleanedSource=null;$("hairMattingButton").disabled=true;$("fileMeta").textContent=file.name+" • "+state.width.toLocaleString("id-ID")+" × "+state.height.toLocaleString("id-ID")+" px";$("previewStatus").textContent="Foto siap diedit";showEditor();toast("Foto diunggah. AI menghapus background dan akan merapikan detail rambut otomatis…");await removeBackground({autoRefine:false})}catch(e){toast(e.message)}}
$("photoInput").addEventListener("change",e=>loadMainPhoto(e.target.files[0]));
$("dropzone").addEventListener("dragover",e=>{e.preventDefault();$("dropzone").classList.add("dragover")});
$("dropzone").addEventListener("dragleave",()=>$("dropzone").classList.remove("dragover"));
$("dropzone").addEventListener("drop",e=>{e.preventDefault();$("dropzone").classList.remove("dragover");loadMainPhoto(e.dataTransfer.files[0])});
$("removeButton").addEventListener("click",removeBackground);
document.querySelectorAll(".swatch").forEach(b=>{b.addEventListener("pointerenter",()=>setBackground(b.dataset.bg));b.addEventListener("click",()=>setBackground(b.dataset.bg))});
bindLivePicker($("saturationPicker"),"saturation");bindLivePicker($("huePicker"),"hue");applyLivePicker();
$("backgroundInput").addEventListener("change",async e=>{const file=e.target.files[0];if(!file)return;try{const loaded=await imageFromFile(file);state.backgroundImage=loaded.img;$("backgroundImageMeta").textContent="Background: "+file.name;setBackground(loaded.img);toast("Background gambar berhasil diterapkan.")}catch(err){toast(err.message)}});
$("eraseMode").addEventListener("click",()=>{state.mode="erase";$("eraseMode").classList.add("active");$("restoreMode").classList.remove("active")});
$("restoreMode").addEventListener("click",()=>{state.mode="restore";$("restoreMode").classList.add("active");$("eraseMode").classList.remove("active")});
$("brushSize").addEventListener("input",e=>{state.brushSize=Number(e.target.value);$("brushValue").textContent=state.brushSize+" px"});let edgeTuneTimer=null;function scheduleEdgeTune(){clearTimeout(edgeTuneTimer);edgeTuneTimer=setTimeout(()=>{if(!state.baseMask)return;state.mask=createCanvas(state.width,state.height);state.mask.getContext("2d").drawImage(state.baseMask,0,0);applyEdgeCleanup();decontaminateEdgeColors();render()},180)}$("edgeCleanup").addEventListener("input",e=>{state.edgeCleanup=Number(e.target.value);$("edgeCleanupValue").textContent=state.edgeCleanup+"%";scheduleEdgeTune()});$("decontaminate").addEventListener("input",e=>{state.decontaminate=Number(e.target.value);$("decontaminateValue").textContent=state.decontaminate+"%";scheduleEdgeTune()});
$("hairMattingButton").addEventListener("click",refineHairMatting);$("resetMask").addEventListener("click",resetMask);
brush.addEventListener("pointerdown",onBrushDown);brush.addEventListener("pointermove",onBrushMove);brush.addEventListener("pointerup",onBrushUp);brush.addEventListener("pointercancel",onBrushUp);
$("exportPng").addEventListener("click",exportPng);$("exportJpg").addEventListener("click",exportJpg);$("exportPdf").addEventListener("click",exportPdf);
$("fitButton").addEventListener("click",()=>{if(state.source){showEditor();toast("Pratinjau disesuaikan ke area.")}});
$("resetView").addEventListener("click",()=>{if(state.source)render()});
// Biarkan browser menampilkan prompt instalasi PWA secara native.
if("serviceWorker"in navigator)window.addEventListener("load",()=>navigator.serviceWorker.register("./sw.js").catch(e=>console.warn("Service worker gagal didaftarkan:",e)));
enableExports();