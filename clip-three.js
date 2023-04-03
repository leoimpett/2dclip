{/* <script> */}
/////////////
//  STEP 1 //
/////////////

var workersInitialised = false; // will be set to true after workers are initialized


// var maxNImages = 6000;

// doAll function
function doAll() {
  searchBtn.disabled = true;
  // disable changeFolder
  // changeFolder.disabled = true;





  // initialize workers
  if (!workersInitialised){initializeWorkers();}
  

  // pick directory
    pickDirectory({source:'local'});
    console.log("directory picked")

}


// first we need to download the models and initialize the workers 
// window.MODEL_NAME = "clip_vit_32";
window.MODEL_NAME = "clip_vit_32_uint8"
window.modelData = {
  clip_vit_32: {
    image: {
      modelUrl: (quantized) => `https://huggingface.co/rocca/openai-clip-js/resolve/main/clip-image-vit-32-${quantized ? "uint8" : "float32"}.onnx`,
      embed: async function(blob, session) {
        let rgbData = await getRgbData(blob);
        const feeds = {input: new ort.Tensor('float32', rgbData, [1,3,224,224])};
        const results = await session.run(feeds);
        const embedVec = results["output"].data; // Float32Array
        return embedVec;
      }
    },
    text: {
      modelUrl: (quantized) => `https://huggingface.co/rocca/openai-clip-js/resolve/main/clip-text-vit-32-${quantized ? "uint8" : "float32-int32"}.onnx`,
      embed: async function(text, session) {
        if(!window.textTokenizerClip) {
          let Tokenizer = (await import("https://deno.land/x/clip_bpe@v0.0.6/mod.js")).default;
          window.textTokenizerClip = new Tokenizer(); 
        }
        let textTokens = window.textTokenizerClip.encodeForCLIP(text);
        textTokens = Int32Array.from(textTokens);
        const feeds = {input: new ort.Tensor('int32', textTokens, [1, 77])};
        const results = await session.run(feeds);
        return [...results["output"].data];
      },
    }
  },
  lit_b16b: {
    image: {
      modelUrl: () => 'https://huggingface.co/rocca/lit-web/resolve/main/embed_images.onnx',
      embed: async function(blob, session) {
        
        // TODO: Maybe remove tf from this code so you can remove the whole tfjs dependency
        blob = await bicubicResizeAndCenterCrop(blob);
        let inputImg = new Image();
        await new Promise(r => inputImg.onload=r, inputImg.src=URL.createObjectURL(blob));
        let img = tf.browser.fromPixels(inputImg);
        img = tf.sub(tf.div(tf.expandDims(img), 127.5), 1);
        let float32RgbData = img.dataSync();
        
        const feeds = {'images': new ort.Tensor('float32', float32RgbData, [1,224,224,3])};
        const results = await session.run(feeds);
        return results["Identity_1:0"].data;
      },
    },
    text: {
      modelUrl: () => 'https://huggingface.co/rocca/lit-web/resolve/main/embed_text_tokens.onnx',
      embed: async function(text, session) {
        // Here we use a custom tokenizer that is not part of the model
        if(!window.bertTextTokenizerLit) {
          window.bertTextTokenizerLit = await import("./bert-text-tokenizer.js").then(m => new m.BertTokenizer());
          await window.bertTextTokenizerLit.load();
        }
        let textTokens = window.bertTextTokenizerLit.tokenize(text);
        textTokens.unshift(101); // manually put CLS token at the start
        textTokens.length = 16;
        textTokens = [...textTokens.slice(0, 16)].map(e => e == undefined ? 0 : e); // pad with zeros to length of 16
        textTokens = Int32Array.from(textTokens);
        const feeds = {'text_tokens': new ort.Tensor('int32', textTokens, [1,16])};
        const results = await session.run(feeds);
        return [...results["Identity_1:0"].data];
      }
    }
  },
};

let imageWorkers = [];
let onnxImageSessions = [];
let onnxTextSession;
let textTokenizer;
async function initializeWorkers() {

  workersInitialised = true; 

  console.log("initialising workers")


  // show downloadingProgressBars
  document.getElementById("downloadingProgressBars").style.display = "block";
  console.log('showing progressbar')


  // initWorkersBtn.disabled = true;
  // numThreadsEl.disabled = true;
  
  let useQuantizedModel = false;
  
  if(MODEL_NAME.endsWith("_uint8")) {
    MODEL_NAME = MODEL_NAME.replace(/_uint8$/g, "");
    useQuantizedModel = true;
  }
  
  let imageOnnxBlobPromise = downloadBlobWithProgress(window.modelData[MODEL_NAME].image.modelUrl(useQuantizedModel), function(e) {
    let ratio = e.loaded / e.total;
    imageModelLoadingProgressBarEl.value = ratio;
    imageModelLoadingMbEl.innerHTML = Math.round(ratio*e.total/1e6)+" MB";
  });

  let textOnnxBlobPromise = downloadBlobWithProgress(window.modelData[MODEL_NAME].text.modelUrl(useQuantizedModel), function(e) {
    let ratio = e.loaded / e.total;
    textModelLoadingProgressBarEl.value = ratio;
    textModelLoadingMbEl.innerHTML = Math.round(ratio*e.total/1e6)+" MB";
  });

  let [imageOnnxBlob, textOnnxBlob] = await Promise.all([imageOnnxBlobPromise, textOnnxBlobPromise])
  console.log("Blob sizes: ", imageOnnxBlob.size, textOnnxBlob.size);

  let imageModelUrl = window.URL.createObjectURL(imageOnnxBlob);
  let textModelUrl = window.URL.createObjectURL(textOnnxBlob);

  // console.log("URLs: ", imageModelUrl, textModelUrl);
  
  // let numImageWorkers = Number(numThreadsEl.value);
  let numImageWorkers = 4;
  
  // Inference latency is about 5x faster with wasm threads, but this requires these headers: https://web.dev/coop-coep/ I'm using this as a hack (in enable-threads.js) since Github pages doesn't allow setting headers: https://github.com/gzuidhof/coi-serviceworker
  if(self.crossOriginIsolated) {
    ort.env.wasm.numThreads = Math.ceil(navigator.hardwareConcurrency / numImageWorkers) / 2; // divide by two to utilise only half the CPU's threads because trying to use all the cpu's threads actually makes it slower
  }

  // workerInitProgressBarEl.max = numImageWorkers + 2; // +2 because of text model and bpe library
  
  let imageModelExecutionProviders = ["wasm"]; // webgl is not compatible with this model (need to tweak conversion data/op types)

  for(let i = 0; i < numImageWorkers; i++) {
    let session = await ort.InferenceSession.create(imageModelUrl, { executionProviders: imageModelExecutionProviders }); 
    onnxImageSessions.push(session);
    imageWorkers.push({
      session,
      busy: false,
    });
    // workerInitProgressBarEl.value = Number(workerInitProgressBarEl.value) + 1;
  }
  console.log("Image model loaded.");

  onnxTextSession = await ort.InferenceSession.create(textModelUrl, { executionProviders: ["wasm"] }); // webgl is not compatible with this model (need to tweak conversion data/op types)
  console.log("Text model loaded.");
  // workerInitProgressBarEl.value = Number(workerInitProgressBarEl.value) + 1;

  window.URL.revokeObjectURL(imageModelUrl);
  window.URL.revokeObjectURL(textModelUrl);

  window.vips = await Vips(); // for bicubicly resizing images (since that's what CLIP expects)
  window.vips.EMBIND_AUTOMATIC_DELETELATER = false;

  // workerInitProgressBarEl.value = Number(workerInitProgressBarEl.value) + 1;

  // disableCtn(initCtnEl);
  // enableCtn(pickDirCtnEl);

  // hide searchSpinner
  document.getElementById("searchSpinner").style.display = "none";


  // hide downloadingProgressBars
  document.getElementById("downloadingProgressBars").style.display = "none";
  console.log('hiding progressbar')

  


}


/////////////
//  STEP 2 //
/////////////
let directoryHandle;
let embeddingsFileHandle;
let embeddings;
let dataSource;
async function pickDirectory(opts={}) {
  dataSource = opts.source;
   
  if(dataSource === "local") {
    if(!window.showDirectoryPicker) return alert("Your browser does not support some modern features (specifically, File System Access API) required to use this web app. Please try updating your browser, or switching to Chrome, Edge, or Brave.");
    directoryHandle = await window.showDirectoryPicker();
    embeddingsFileHandle = await directoryHandle.getFileHandle(`${window.MODEL_NAME}_embeddings.tsv`, {create:true});
    

  }
  
  // let redditEmbeddingsBlob;
  // if(dataSource === "reddit") {
  //   if(window.MODEL_NAME !== "clip_vit_32") return alert("Sorry, there are only pre-computed Reddit image embeddings for the CLIP ViT-B/32 model at the moment.");
  //   if(!removeRedditNsfwEl.checked && !confirm("Are you sure you'd like to see NSFW Reddit images?")) return;
  //   if(removeRedditNsfwEl.checked) alert("Note that NSFW images are filtered from Reddit using CLIP, and CLIP can make mistakes, so some NSFW images may still be shown.");
      
  //   // pickDirectoryBtn.disabled = true;
  //   // useRedditImagesBtn.disabled = true;
  //   // useRedditImagesBtn.textContent = "Loading...";
  //   // redditLoadProgressCtn.style.display = "";
    
  //   redditEmbeddingsBlob = await downloadBlobWithProgress("https://huggingface.co/datasets/rocca/top-reddit-posts/resolve/main/clip_embeddings_top_50_images_per_subreddit.tsv.gz", function(e) {
  //     let ratio = e.loaded / e.total;
  //     redditProgressBarEl.value = ratio;
  //     redditProgressMbEl.innerHTML = Math.round(ratio*213)+" MB";
  //   });
  // }


  
  try {
    // existingEmbeddingsProgressCtn.style.display = "";
    
    embeddings = {};
    let file, opts;
    if(dataSource === "local") {
      file = await embeddingsFileHandle.getFile();
      opts = {};
    }
    if(dataSource === "online") {
      console.log("Trying to pick from online - but in the wrong place. Something's gone wrong...")
      // file = redditEmbeddingsBlob;
      // opts = {decompress:"gzip"};
      // add this back in for producton:
      file = await embeddingsFileHandle.getFile();
      opts = {};
    }
    
    let i = 0;
    for await (let line of makeTextFileLineIterator(file, opts)) {
      if(!line || !line.trim()) continue; // <-- to skip final new line (not sure if this is needed)
      // see how long line.split("\t") is 
      splitline = line.split("\t");
      if (splitline.length == 2) {
        let [filePath, embeddingVec] = splitline
      // console.log([filePath, embeddingVec])
      embeddings[filePath] = JSON.parse(embeddingVec);
      i++;
      }

    //   if(i % 1000 === 0) {
        // existingEmbeddingsLoadedEl.innerHTML = i;
        // await sleep(10);
    //   }
    }
  } catch(e) {
    // embeddings = undefined;
    console.log("No existing embedding found, or the embeddings file was corrupted:", e);
    // existingEmbeddingsProgressCtn.style.display = "none";
  }
  


    // hide #step1
    document.getElementById("step1").style.display = "none";
    // show #step2
    document.getElementById("step2").style.display = "block";
    // Set step2 class to hover 
    document.getElementById("step2").classList.add("hover");
    // Set timeout to remove hover class 
    setTimeout(function(){
      document.getElementById("step2").classList.remove("hover");
    }, 2000);




  // show searchSpinner
  document.getElementById("searchSpinner").style.display = "block";

  

    //  The end of pickdirectory - is always going to be....
    // Wait until window.vips is defined
    while (window.vips === undefined) {
      // console.log("waiting for vips")
      await sleep(100);
    }

    // Wait until embeddingsFileHandle is resolved

    await embeddingsFileHandle;

    if(dataSource === "local") {
      computeImageEmbeddings() // <-- this is the end of pickdirectory
    }

}


// STEP 2.5 - picking an existing embeddings file
async function getWebDataset(url) {


  // TODO - precalculate the TSNE in the gzipped file - would make things much quicker. 

  dataSource = "online"
  if (!workersInitialised){initializeWorkers();}


  const response = await fetch(url);
const jsonString = await response.text();

const onlineData = JSON.parse(jsonString);

  

// the data is jsonData[key].embeddings and jsonData[key] metadata - split into two dictionaries
// for each key in jsonData, add to embeddings and metadata dictionaries
embeddings = {}
metadata = {}
for (const [key, value] of Object.entries(onlineData)) {
    embeddings[key] = value.embeddings
    metadata[key] = value.metadata
}


//   webDataEmbeddingBlob = await downloadBlobWithProgress( url, function(e){console.log(e.loaded/e.total)})
//   file = webDataEmbeddingBlob;
// //   if url ends in .gz then decompress
// if(url.endsWith(".gz")) {
//   opts = {decompress:"gzip"};
// }
// else {
//     opts = {};
// }

//   embeddings = {};
//   metadata = {};
//   let i = 0;

//   for await (let line of makeTextFileLineIterator(file, opts)) {
//       if(!line || !line.trim()) continue; // <-- to skip final new line (not sure if this is needed)
//       splitline = line.split("\t");
//         let [filePath, embeddingVec, catalog] = splitline
//       embeddings[filePath] = JSON.parse(embeddingVec);
//       metadata[filePath] = JSON.parse(catalog);
//       i++;

      
//     }
//     // when finished, console log embeddings
//     // console.log(embeddings)
    console.log("Loaded precalculated embeddings ")


    // hide #step1
    document.getElementById("step1").style.display = "none";
    // show #step2
    document.getElementById("step2").style.display = "block";
    // Set step2 class to hover 
    document.getElementById("step2").classList.add("hover");
    // Set timeout to remove hover class 
    setTimeout(function(){
      document.getElementById("step2").classList.remove("hover");
    }, 2000);


    // hide localImagePanel
    // document.getElementById("localImagePanel").style.display = "none";
    

    searchSort();


}



/////////////
//  STEP 3 //
/////////////
let totalEmbeddingsCount = 0;
let imagesEmbedded;
let recentEmbeddingTimes = []; // how long each embed took in ms, newest at end
let recomputeAllEmbeddings;
let imagesBeingProcessedNow = 0; 
let needToSaveEmbeddings = false;
async function computeImageEmbeddings() {

  // show computedEmbeddings

  console.log("Computing image embeddings...");
  imagesEmbedded = 0;
  totalEmbeddingsCount = Object.keys(embeddings).length;

  console.log("Got this number of embeddings: " + totalEmbeddingsCount)
  onlyEmbedNewImages = 1;

  recomputeAllEmbeddings = !onlyEmbedNewImages;
  let gotSomeExistingEmbeddings = totalEmbeddingsCount > 0;

  // Try: if not gotSomeExistingEmbeddings, then force recomputeAllEmbeddings to be true
  if (!gotSomeExistingEmbeddings) {
    console.log("forcing recompute")
    recomputeAllEmbeddings = true;
  }
  
  if(onlyEmbedNewImages && gotSomeExistingEmbeddings) {
    // preexistingEmbeddingsEl.display = "block";
    // preexistingEmbeddingsEl.innerHTML = `Loaded ${Object.keys(embeddings).length} preprocessed images.`; 
    // hide computedEmbeddings 
    // document.getElementById("computedEmbeddings").style.display = "none";
  }
  else {
    // preexistingEmbeddingsEl.display = "none";
    // document.getElementById("computedEmbeddings").style.display = "block";
  }

  if(recomputeAllEmbeddings || !gotSomeExistingEmbeddings) {
    embeddings = {}; // <-- maps file path (relative to top/selected directory) to embedding
  }

  // console.log(recomputeAllEmbeddings, gotSomeExistingEmbeddings, Object.keys(embeddings).length)
  
  try {
    await recursivelyProcessImagesInDir(directoryHandle);
    await saveEmbeddings();
  } catch(e) {
    console.error(e);
    alert(e.message);
  }

  // disableCtn(computeEmbeddingsCtnEl);
  // enableCtn(searchCtnEl);

  // hide loading spinner
  document.getElementById("searchSpinner").style.display = "none";

  console.log("Done computing image embeddings.");

  searchSort();

}


async function recursivelyProcessImagesInDir(dirHandle, currentPath="") {


  // console.log(dirHandle, currentPath)
      // image count first!!! 
            let imageCount = 0;

          // Count the number of image files
          for await (let [name, handle] of dirHandle) {
            const {kind} = handle;
            let path = `${currentPath}/${name}`;
           if(path.includes("/thumbnails")) continue;
            if (handle.kind === 'directory') {
              imageCount += await recursivelyProcessImagesInDir(handle, path);
            } else {
              // make lower case copy of path
              let pathLower = path.toLowerCase();

              let isImage = /\.(png|jpg|jpeg|webp|JPEG|JPG)$/.test(pathLower);
              if(!isImage) continue;

              imageCount++;
            }
          }

          // Print the total number of image files
          // console.log(`Total number of image files: ${imageCount}`);

          // If imageCount > maxNImages, then alert
          // if (imageCount > maxNImages) {
          //   alert(`You have selected a directory with ${imageCount} images. This is more than the maximum number of images recommended (1000).`);
          // }


          // set innerhtml of totalNumberImages to imageCount 
          // document.getElementById("totalNumberImages").innerHTML = imageCount;



  for await (let [name, handle] of dirHandle) {
    const {kind} = handle;
    let path = `${currentPath}/${name}`;
      // console.log(path)
      // ignore folder ./thumbnails/
      if(path.includes("/thumbnails")) continue;

    if ((handle.kind === 'directory') ) {
      await recursivelyProcessImagesInDir(handle, path);
    } else {
      // make lower case copy of path
      let pathLower = path.toLowerCase();


      let isImage = /\.(png|jpg|jpeg|webp|JPEG|JPG)$/.test(pathLower);
      if(!isImage) continue;

      // console.log("Processing image:", path)

      let alreadyGotEmbedding = !!embeddings[path];

      // console.log("Alreadygotembedding:",alreadyGotEmbedding)
      // console.log("Recompute:", recomputeAllEmbeddings)
      // console.log("needToSaveEmbeddings:", needToSaveEmbeddings)

      if(alreadyGotEmbedding && !recomputeAllEmbeddings) continue;
      
      if(needToSaveEmbeddings) {
        await saveEmbeddings();
        needToSaveEmbeddings = false;
      }
        
      while(imageWorkers.filter(w => !w.busy).length === 0) await sleep(1);
      
      let worker = imageWorkers.filter(w => !w.busy)[0];
      worker.busy = true;
      imagesBeingProcessedNow++;
      
      // if (Object.keys(embeddings).length >= maxNImages){continue}
      

      (async function() {

        if (Object.keys(embeddings).length >= imageCount){
          return;
        }

        // let startTime = Date.now();
        

        // try
        try{
        let blob = await handle.getFile();
        const embedVec = await modelData[MODEL_NAME].image.embed(blob, worker.session);
        // TODO - we can probably embed the path rather than the blob. This way we can use the embed function
        // to save the thumbnails, rather than computing all the the thumbs twice (as 224 for embedding and as 256 for the rendering). 
        // This is only an advantage when you have large images (otherwise you dont do the second step)

        // What we need to do:
        // define directoryHandleThumb  above the loop
        // define thumbnailPath within the loop
        // pass both directoryHandleThumb and thumbnailPath to image.embed function
        // image.embed then passes it to bicubicResizeAndCenterCrop() 
        // NB - that makes a centre crop (224x224), which is what we want for embedding but not for thumbnail display
        // So within the same canvas we might first draw a 256x256 image to save as a thumbnail, without cropping, then resize and crop and pass the blob back. 
        // Is this going to be any faster? It's not clear. We are at least limiting the number of read/write operations. 
        // However, we might do an alternative approach - leave the 4 workers to do the embedding, and start doing the thumbnailing in the meantime. 
        // By this point we do, after all, already have the paths to all images - see where imageCount is defined above. 

        embeddings[path] = [...embedVec];
        worker.busy = false;

        imagesEmbedded++;
        totalEmbeddingsCount++;
        }
        catch(e){
          console.log(e)
          console.log("Failed to process image ", path)
        worker.busy = false;
        }

        computeEmbeddingsProgressEl = document.getElementById("computeEmbeddingsProgress");
        computeEmbeddingsLoadingProgressBarEl = document.getElementById("computeEmbeddingsLoadingProgressBarEl");
        computeEmbeddingsText = document.getElementById("computeEmbeddingsText");
        

        // console.log(`Embedded ${Object.keys(embeddings).length} images in ${Date.now() - startTime} ms`);

        // computeEmbeddingsProgressEl.innerHTML = Object.keys(embeddings).length;

        // Update computeEmbeddingsLoadingProgressBarEl with the ratio of imagesEmbedded to imageCount
        if(imageCount){
        computeEmbeddingsLoadingProgressBarEl.value = Object.keys(embeddings).length / imageCount;
        // Make sure that the progress bar is visible
        computeEmbeddingsProgressEl.style.display = "block";
        computeEmbeddingsText.innerHTML = `Encoding images (${Object.keys(embeddings).length} of ${imageCount})`;
    
      }

        // If the ratio is not one, set step2 to :hover. Else remove :hover
        // maybe get rid of this
        // if ((Object.keys(embeddings).length < imageCount)&&(Object.keys(embeddings).length < maxNImages)){
        //   document.getElementById("step2").classList.add("hover");
        // } else {
        //   document.getElementById("step2").classList.remove("hover"); 
        // }

        

        
        let saveInterval = totalEmbeddingsCount > 50_000 ? 10_000 : 1000; // since saves take longer if there are lots of embeddings
        if(imagesEmbedded % saveInterval === 0) {
          needToSaveEmbeddings = true;
        }
        
        // recentEmbeddingTimes.push(Date.now()-startTime);
        // if(recentEmbeddingTimes.length > 100) recentEmbeddingTimes = recentEmbeddingTimes.slice(-50);
        // // if(recentEmbeddingTimes.length > 10) computeEmbeddingsSpeedEl.innerHTML = Math.round(recentEmbeddingTimes.slice(-20).reduce((a,v) => a+v, 0)/20);

        // // Compute the expected time left
        // let expectedTimeLeft = Math.round((imageCount - Object.keys(embeddings).length) * recentEmbeddingTimes.slice(-20).reduce((a,v) => a+v, 0)/20);
        // // convert to minutes and seconds
        // const expectedTimeString = `${Math.floor(expectedTimeLeft / 60000).toString().padStart(2, '0')}:${Math.floor((expectedTimeLeft % 60000) / 1000).toString().padStart(2, '0')}`;

        // if(recentEmbeddingTimes.length > 10) computeEmbeddingsSpeedEl.innerHTML = expectedTimeString;

        


        imagesBeingProcessedNow--;
      })();


    }
  }
  while(imagesBeingProcessedNow > 0) await sleep(10);
}


/////////////
//  STEP 4 //
/////////////

isRendered = false; 



  // Change this to only fire once - on loading the directory. Handlers can then change the x-y coordinates of the images based on new axis values. 
async function searchSort() {


  searchBtn.disabled = true;
  // searchSpinner show
  document.getElementById("searchSpinner").style.display = "block";
  
  
  // resultsEl.innerHTML = "Loading...";
  await sleep(50);


  let resultHtml = "";
  let numResults = 0;
  // imageResults = [];



    // Then do UMAP stuff
    dataArray = [];
    orderedPath = [];
  for(let [path, embedding] of Object.entries(embeddings)) {
    // similarities[path] = cosineSimilarity(searchTextEmbedding, embedding);
    dataArray.push(embedding);
    orderedPath.push(path);
  }

  console.log("calculating umap...");


  // const umap = new UMAP({distanceFn:"cosine"});
  // Cosine distance doesn't seem to be working - check again some other time. 
//   const umap = new UMAP({minDist:.1});
  const umap = new UMAP({minDist:.1, nEpochs:300});

  document.getElementById("UMAPProgressBar").style.display = "block";

const umapPromise = umap.fitAsync(dataArray, epochNumber => {
  // check progress and give user feedback, or return `false` to stop

    // display UMAPProgressBar 
    UMAPProgressBarEl.value = epochNumber / 300;
    // console.log("Epoch number: ", epochNumber);

  
});

console.log("calculating umap and loading embeddings in parallel...");

const imageResultsPromise = loadThumbnails(embeddings);

const [umap_embedding, imageResults] = await Promise.all([umapPromise, imageResultsPromise]);

// Get rid of UMAPProgressBar
document.getElementById("UMAPProgressBar").style.display = "none";

console.log("done calculating umap and loading embeddings");

// Set d.score and d.score2 to umap_embedding[i].x and umap_embedding[i].y
for (let i = 0; i < imageResults.length; i++) {
  ref_path = imageResults[i].path;
  umap_index = orderedPath.indexOf(ref_path);
  imageResults[i].score = umap_embedding[umap_index][0];
  imageResults[i].score2 = umap_embedding[umap_index][1];
}


  imageData = normalizeGrid(imageResults);

  console.log("loaded thumbnails - rendering grid...");

        // add checkbox forceDirected 
        const forceDirected = d3.select("#forceDirected")
        forceDirected.on("change", function(event) {
          if (event.target.checked) {
            console.log("checked")

            for (let i = 0; i < scene.children.length; i++) {
        const sprite = scene.children[i];
        // change the sprite's position here
        // if the sprite has userData.force_x defined...
        if(sprite.userData.force_x){
          sprite.position.set(  sprite.userData.force_x, sprite.userData.force_y);

          // console.log(sprite.userData.force_x, sprite.userData.force_y)
        }
      } 

      animate();
            

          } else {
            // console.log("unchecked")
            // images.attr("x", d => d.orig_x)
            //       .attr("y", d => d.orig_y);

            for (let i = 0; i < scene.children.length; i++) {
        const sprite = scene.children[i];
        // if the sprite has userData.orig_x defined...
        if(sprite.userData.orig_x){
        sprite.position.set(  sprite.userData.orig_x, sprite.userData.orig_y);
        }

      } 

      animate();
          }
        });

        // Add a similar callback for resetUmap button
        const resetUmap = d3.select("#resetUmap")
        resetUmap.on("click",function (event) {

            // set x-axis-label and y-axis-label to ""
            document.getElementById("x-axis-label").innerHTML = "";
            document.getElementById("y-axis-label").innerHTML = "";

            // Check that it's not already on UMAP view. Select the first sprite...
            const sprite = scene.children[0];
            // if the sprite has force_x === UMAP_x then it's already on UMAP view
            if (sprite.userData.force_x === sprite.userData.UMAP_x) {
                console.log("Already on UMAP view")
                return;
            }

            //     sprite.userData.UMAP_x  
            // sprite.userData.UMAP_y  
            // sprite.userData.UMAP_orig_x 
            // sprite.userData.UMAP_orig_y  
            // Now set force_x and force_y to UMAP_x and UMAP_y
            
            for (let i = 0; i < scene.children.length; i++) {
                const sprite = scene.children[i];
            sprite.userData.force_x = sprite.userData.UMAP_x;
            sprite.userData.force_y = sprite.userData.UMAP_y;
            sprite.userData.orig_x = sprite.userData.UMAP_orig_x;
            sprite.userData.orig_y = sprite.userData.UMAP_orig_y;
            }
            
            console.log(document.getElementById("forceDirected").checked)
            if (document.getElementById("forceDirected").checked) {
            
                for (let i = 0; i < scene.children.length; i++) {
            const sprite = scene.children[i];
            // change the sprite's position here
            // if the sprite has userData.force_x defined...
            if(sprite.userData.force_x){
              sprite.position.set(  sprite.userData.force_x, sprite.userData.force_y);
            
              // console.log(sprite.userData.force_x, sprite.userData.force_y)
            }
            } 
            
            animate();
                
            
              } else {
                // console.log("unchecked")
                // images.attr("x", d => d.orig_x)
                //       .attr("y", d => d.orig_y);
            
                for (let i = 0; i < scene.children.length; i++) {
            const sprite = scene.children[i];
            // if the sprite has userData.orig_x defined...
            if(sprite.userData.orig_x){
            sprite.position.set(  sprite.userData.orig_x, sprite.userData.orig_y);
            }
            
            } 
            
            animate();
              }
            });


        gridUpdating = false;
        
        const updateGridButton = d3.select("#searchBtn")
        updateGridButton.on("click", function(event) {
          updateGrid();
        });
        function handleKeyUp(event) {
          if (event.which === 13) {
            // updateGrid();
            // click updateGridButton
            updateGridButton.node().click();
          }
        }

        document.querySelector('#searchTextEl').addEventListener('keyup', handleKeyUp);

         document.querySelector('#searchTextEl2').addEventListener('keyup', handleKeyUp);

        //  Set searchtext elements to blank
        document.getElementById("searchTextEl").value = "";
        document.getElementById("searchTextEl2").value = "";
       


        // Add "Plot" button event listener
       async function updateGrid() {


          // X axis label
          const xAxisLabel = document.querySelector("#x-axis-label");
          if (xAxisLabel) {
            xAxisLabel.innerHTML = searchTextEl.value + " →";
            xAxisLabel.className = "axis-label";
          }

          // Y axis label
          const yAxisLabel = document.querySelector("#y-axis-label");
          if (yAxisLabel) {
            yAxisLabel.innerHTML = searchTextEl2.value + " →";
            yAxisLabel.className = "axis-label";
          }

          if (gridUpdating) return;

          gridUpdating=true; 
          // grey out the button
          searchBtn.disabled = true;
          // searchSpinner show
          document.getElementById("searchSpinner").style.display = "inline-block";


          //  // X axis
          //  let searchTextEmbedding = await modelData[MODEL_NAME].text.embed(searchTextEl.value, onnxTextSession);
          // // Y axis
          // let searchTextEmbedding2 = await modelData[MODEL_NAME].text.embed(searchTextEl2.value, onnxTextSession);

          // let similarities = {};
          // let similarities2 = {};
          // for(let [path, embedding] of Object.entries(embeddings)) {
          //   similarities[path] = cosineSimilarity(searchTextEmbedding, embedding);
          //   similarities2[path] = cosineSimilarity(searchTextEmbedding2, embedding);
          // }
          // let similarityEntries = Object.entries(similarities);
          // let similarityEntries2 = Object.entries(similarities2);


          // X axis
          const searchTextEmbeddingPromise = modelData[MODEL_NAME].text.embed(searchTextEl.value, onnxTextSession);

          // Y axis
          const searchTextEmbedding2Promise = modelData[MODEL_NAME].text.embed(searchTextEl2.value, onnxTextSession);

          // Compute cosine similarities in parallel
          const embeddingsEntries = Object.entries(embeddings);
          const similaritiesPromises = embeddingsEntries.map(([path, embedding]) => {
            return Promise.all([searchTextEmbeddingPromise, searchTextEmbedding2Promise]).then(([searchTextEmbedding, searchTextEmbedding2]) => {
              const similarity = cosineSimilarity(searchTextEmbedding, embedding);
              const similarity2 = cosineSimilarity(searchTextEmbedding2, embedding);
              return [path, similarity, similarity2];
            });
          });


          // Wait for all cosine similarities to be computed
          const similarityEntries = await Promise.all(similaritiesPromises);

          // newImageResults = [];
          // // Combine into imageResults
          // for(let [path, score] of similarityEntries) {
          //     let handle = await getFileHandleByPath(path);
          //     let url = URL.createObjectURL(await handle.getFile());
          //     // NB we reverse the y axis as it should point up
          //     let score2 = similarities2[path];
          //     newImageResults.push({url,path, score, score2});
          // }

          // Combine into imageResults
          const imageResultsPromises = similarityEntries.map(async ([path, score, score2]) => {

            // if data source is online, just return the url as path
            if (dataSource == "online") {
                return { url: path, path, score, score2 };
            }

            const handle = await getFileHandleByPath(path);
            const url = URL.createObjectURL(await handle.getFile());
            return { url, path, score, score2 };
          });

          const newImageResults = await Promise.all(imageResultsPromises);

          newImageData = normalizeGrid(newImageResults);
          
          // now update sprite positions
          for (let i = 0; i < newImageData.length; i++) {


            // get the correct sprite - where sprite.userData.path == newImageData[i].path
            const sprite = scene.children.find(sprite => sprite.userData.path == newImageData[i].path);
            // if sprite has position
            if (sprite.position) {
              // console.log("sprite has position")
        // change the sprite's position here
        // console.log(i+","+newImageData[i].x+","+newImageData[i].y)
        sprite.position.set(  newImageData[i].x, newImageData[i].y);
        // update userdata of orig_x orig_y force_x and force_y
        sprite.userData.orig_x = newImageData[i].orig_x;
        sprite.userData.orig_y = newImageData[i].orig_y;
        sprite.userData.force_x = newImageData[i].force_x;
        sprite.userData.force_y = newImageData[i].force_y;
            } 

          }


        // animate
        animate();
        gridUpdating = false;

          // enable the button
          searchBtn.disabled = false;
          // searchSpinner hide
          document.getElementById("searchSpinner").style.display = "none";

       resetZoom();

       }






        // Create a new Three.js scene
        var scene = new THREE.Scene();

        console.log("three.js scene created");

        scene.background = new THREE.Color( 1, 1, 1 )


// // // Add a grid
// // Create a grid helper
// const gridHelper = new THREE.GridHelper(10, 10, 0x555555 , 0xeeeeee ); //, 0x000000, 0xffffff);
// // Rotate the grid to face the camera
// gridHelper.rotation.x = Math.PI / 2;
// // Add the grid helper to the scene
// scene.add(gridHelper);


// Create a new Three.js camera
// camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera = new THREE.OrthographicCamera(window.innerWidth / -2, window.innerWidth/2,window.innerHeight/2, window.innerHeight/-2, 4, 6 )
camera.position.z = 5; 
camera.zoom = 30;
camera.updateProjectionMatrix();

// Create a new Three.js renderer and add it to the page
var renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.domElement.style.position = 'absolute';
renderer.domElement.style.top = 0;
renderer.domElement.style.left = 0;
document.body.appendChild(renderer.domElement);


// Create a new instance of OrbitControls
controls = new THREE.OrbitControls(camera, renderer.domElement);
// controls.enableDamping = true;
// controls.dampingFactor = 0.1;
controls.enablePan = true;
controls.enableZoom = true;
// controls.minDistance = .1;
// controls.maxDistance = 50;
// for orthographic camera this is actually min/maxZoom
controls.minZoom = 2;
controls.maxZoom = 5000;
controls.mouseButtons = { 
LEFT: THREE.MOUSE.PAN,
MIDDLE: THREE.MOUSE.DOLLY
}

// Restrict camera movement to 2D panning and zooming
controls.enableRotate = false;
controls.enableKeys = false; // Disable arrow keys and WASD for rotation
controls.minPolarAngle = Math.PI / 2;
controls.maxPolarAngle = Math.PI / 2;

          spritesAdded = 0;
          maxLength = 0.2;
          // // if > 1000 images then set maxLength to 0.05
          // if (imageData.length > 1000) {
          //   maxLength = 0.05;
          // }
          // // if > 3000 images then set maxLength to 0.01
          // if (imageData.length > 3000) {
          //   maxLength = 0.025;
          // }


          texloader = new THREE.TextureLoader()

          // merge textures
          // var textureMerger = new TextureMerger();
          // for (let i = 0; i < imageData.length; i++) {
          //   const url =  imageData[i].url;
          //   textureMerger["texture" + (i + 1)] = loadedTextures[i];
          // }
              

            console.log("spawning sprites ");


          for (let i = 0; i < imageData.length; i++) {
            
const url =  imageData[i].url;
const sprite = new THREE.Sprite();
const texture = texloader.load( url, function() {
// Determine sprite dimensions and aspect ratio
const aspectRatio = texture.image.width / texture.image.height;
let width, height;
if (aspectRatio >= 1) {
  // width longer
  width = maxLength;
  height = maxLength / aspectRatio;
} else {
  // height longer
  width = maxLength * aspectRatio;
  height = maxLength;
}

// // if neither width nor height are equal maxLength, log
// if (width != maxLength && height != maxLength) {
//   console.log("sprite width and height not equal to maxLength");
// }
// This correctly never happens

// Set sprite scale and save original dimensions and aspect ratio in userData
sprite.userData.width = width;
sprite.userData.height = height;
sprite.userData.aspectRatio = aspectRatio;
sprite.scale.set(width, height, 1.0);
// attempt fix as some images are smaller than others
// sprite.scale.set(maxLength,maxLength,1.0);

// set catalog metadata
// if metadata is not null
if (typeof metadata !== 'undefined' && metadata !== null) {
    sprite.userData.metadata = metadata[sprite.userData.path];
}

// The sprite is being created during UMAP, so we can save userData.UMAP_x and userData.UMAP_y and userData.UMAP_orig_x and userData.UMAP_orig_y
sprite.userData.UMAP_x = imageData[i].x;
sprite.userData.UMAP_y = imageData[i].y;
sprite.userData.UMAP_orig_x = imageData[i].orig_x;
sprite.userData.UMAP_orig_y = imageData[i].orig_y;

// free up the image URL
window.URL.revokeObjectURL(url);
});

sprite.material = new THREE.SpriteMaterial({ map: texture, transparent: true });

// Set X and Y coordinates from imageData
sprite.position.x = imageData[i].x;
sprite.position.y = imageData[i].y;
sprite.position.z = 0;

sprite.userData.orig_x = imageData[i].orig_x;
sprite.userData.orig_y = imageData[i].orig_y;

sprite.userData.force_x = imageData[i].x;
sprite.userData.force_y = imageData[i].y;

// set sprite center
sprite.center = new THREE.Vector2(0.5, 0.5);


sprite.userData.path = imageData[i].path;

scene.add(sprite);
spritesAdded++;

// for every 100 sprites added, log in console
if (spritesAdded % 1000 === 0) {
console.log("Sprites added: " + spritesAdded);
}

}

isDragging = false;
isPopup = false;




function bindEvents(state) {
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function onDocumentMouseDown(event) {
isDragging = false;
// Calculate normalized mouse coordinates
mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

// Set raycaster origin and direction
raycaster.setFromCamera(mouse, state.camera);
}

async function onDocumentMouseUp(event) {

// check that the mouse is not over a div 
// if it is, then do not do anything
// Get the element that was clicked
const targetElement = event.target;
// Check if the element or any of its ancestors matches the '#step2' selector
if (targetElement.closest('#step2')) {
// The mouse was clicked over the '#step2' element or one of its descendants
return;
}



// Calculate normalized mouse coordinates
mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
raycaster.setFromCamera(mouse, state.camera);

// console.log("mouse up")
// console.log(mouse.x, mouse.y)

// if isDragging, continue
if (isDragging||isPopup) {
// console.log("isDragging")
return;
}
// else{console.log("is not dragging")}
// Prevent default browser behavior
event.preventDefault();
// Reset dragging flag to false
isDragging = false;
// Find intersected objects
const intersects = raycaster.intersectObjects(state.scene.children, true);
// console.log(intersects)
// Log the URL of the clicked sprite
if (intersects.length > 0 && !isDragging) {


// Find intersected sprite
const sprite = intersects.find((intersect) => intersect.object.isSprite)?.object;
if (sprite) {
  // console.log("sprite found")
  // console.log(sprite)

  // if sprite.userData.catalog defined then use that
  if (sprite.userData.metadata){

    console.log(sprite.userData.metadata)
    console.log(sprite.userData.metadata['large'])
    console.log(sprite.userData.metadata.catalog)

    const imagePopup = document.querySelector("#image-popup");
    imagePopup.style.display = "flex";
    const imagePopupContent = document.querySelector("#image-popup-content");
    const imagePopupText = document.querySelector("#image-popup-text");


    // if sprite.userData.metadata.iiif exists...

    // for now - never do this
    // if you want it, change 'iiifDELETEME' to 'iiif' - but the universal viewer is slow and buggy... 
    if ('iiifDELETEME' in sprite.userData.metadata){
        iiifLink = sprite.userData.metadata.iiif;
        // if it starts with "https://api.fitz.ms/data-distributor", quick fix.....
        if (iiifLink.startsWith("https://api.fitz.ms/data-distributor")){
            console.log("Fitz fix")
            objectID = 'object-' + sprite.userData.metadata.catalog.split('/').pop()
            iiifLink = "https://api.fitz.ms/data-distributor/iiif/" + objectID + "/manifest"
            console.log(iiifLink)
        }
                // add an iframe to imagePopupContent
                imagePopupContent.innerHTML = `<iframe src="https://1p175.csb.app/uv.html#?manifest=${sprite.userData.metadata.iiif}" width="100%" height="100%" frameborder="0" allowfullscreen></iframe>`
    }

    // otherwise - seeing as the Fitz iiif links are broken, just use the catalogue link

    else{
    imagePopupContent.style.backgroundImage = `url(${sprite.userData.metadata.large})`;
    imagePopupText.innerHTML = `<a href="${sprite.userData.metadata.catalog}" target="_blank"> <span class="material-symbols-outlined"> preview </span></a>`;
}



    isPopup=true;
    // open new tab
    // TODO - implement this. And maybe include both hi res image AND link to catalogue? 


  }

  else if (sprite.userData.path) {
    // console.log(sprite.userData);

    // make url from path
    // make sure path isn't a url
    if (sprite.userData.path.startsWith("http")){
        console.log("path is a url")
        url = sprite.userData.path;
    }
    else{
    handle = await getFileHandleByPath(sprite.userData.path); 
    url = URL.createObjectURL(await handle.getFile());
    }


    const imagePopup = document.querySelector("#image-popup");
    imagePopup.style.display = "flex";
    const imagePopupContent = document.querySelector("#image-popup-content");
    imagePopupContent.style.backgroundImage = `url(${url})`;
    isPopup=true;
  }
}



}
}

const imagePopup = document.querySelector('#image-popup');
imagePopup.addEventListener('click', () => {
imagePopup.style.display = 'none';
isPopup=false;
});


// Save the original window.innerWidth and window.innerHeight
// const windowWidthOrig = window.innerWidth;
// const windowHeightOrig = window.innerHeight;


function onDocumentMouseMove(event) {
// If moving, set dragging flag to true
isDragging = true;

// Calculate normalized mouse coordinates
mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

// Set raycaster origin and direction
raycaster.setFromCamera(mouse, state.camera);
}


            function onWindowResize() {

            // Update mouse coordinates when window is resized
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

            // // Update camera aspect ratio
            // camera.aspect = window.innerWidth / window.innerHeight;
            // Update left right top bottom frustrums
            camera.left = -window.innerWidth / 2;
            camera.right = window.innerWidth / 2;
            camera.top = window.innerHeight / 2;
            camera.bottom = -window.innerHeight / 2;

            camera.updateProjectionMatrix();

            // Resize renderer to match new window dimensions
            renderer.setSize(window.innerWidth, window.innerHeight);

            }



function onCameraChange() {
// Update raycaster camera when camera changes position or zoom level
raycaster.camera = state.camera;
}

// Bind event listeners
window.addEventListener('mousedown', onDocumentMouseDown, false);
window.addEventListener('mouseup', onDocumentMouseUp, false);
window.addEventListener('mousemove', onDocumentMouseMove, false);
window.addEventListener('resize', onWindowResize, false);

// Set raycaster camera
raycaster.camera = state.camera;

// Listen for camera changes and update raycaster camera
state.camera.addEventListener('change', onCameraChange);
}


bindEvents({ scene, camera });


      // Render the scene
      function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}





animate();

searchBtn.disabled = false;
// #searchSpinner hide 
document.getElementById("searchSpinner").style.display = "none";

isRendered = true;

}


// Hide toolbar when not in use
const checkbox = document.getElementById('hideToolbar')

    checkbox.addEventListener('change', (event) => {
      if (event.currentTarget.checked) {
        // alert('checked');
        document.documentElement.style.setProperty('--hide-opacity', '0');
      } else {
        // alert('not checked');
        // Remove "extrahide" class to step2
        document.documentElement.style.setProperty('--hide-opacity', '0.3');
      }
    })




/////////////////////////////
//  FUNCTIONS / UTILITIES  //
/////////////////////////////



function resetZoom(){
    controls.reset();
}


function normalizeGrid(imageResults){


  var imageData = [];

  // Ideal density is roughly 100 images for a 2x2 grid - thus 25 images / unit. 
  // We will scale the grid to fit the number of images we have.
  const idealDensity = 15;
  const density = imageResults.length / idealDensity;
  const scale = Math.sqrt(density);
  console.log("Scale: "+ scale + ", Density: " + density + ", Ideal Density: " + idealDensity)




  // Get the minimum and maximum values of score and score2
    const [minScore, maxScore] = d3.extent(imageResults, d => d.score);
    const [minScore2, maxScore2] = d3.extent(imageResults, d => d.score2);


    console.log("Range at start " + minScore + " " + maxScore + " " + minScore2 + " " + maxScore2)

    // Create a linear scale for score and score2
    const scaleScore = d3.scaleLinear()
      .domain([minScore, maxScore])
      .range([-scale, scale]);

    const scaleScore2 = d3.scaleLinear()
      .domain([minScore2, maxScore2])
      .range([-scale, scale]);



    for (let i = 0; i < imageResults.length; i++) {
    imageData.push({
      x: scaleScore(imageResults[i].score)  ,
      y: (scaleScore2(imageResults[i].score2) ),
      score: imageResults[i].score,
      score2: imageResults[i].score2,
      url: imageResults[i].url,
      path: imageResults[i].path,
      // also save the title as the end bit of the url
      // title: imageResults[i].url.split("/").slice(-1)[0],
      radius:20
    });
  }


    // Try force directed layout

    // centre the points so the median is at the centre
    const xMedian = d3.median(imageData, d => d.x);
    const yMedian = d3.median(imageData, d => d.y);
    for (let i = 0; i < imageData.length; i++) {
      imageData[i].x -= xMedian;
      imageData[i].y -= yMedian;
    }

    // Save the old (x,y) pairs as orig_x and orig_y
    for (let i = 0; i < imageData.length; i++) {
      imageData[i].orig_x = imageData[i].x;
      imageData[i].orig_y = imageData[i].y;
    }



    const collisionDistance = 0.12;

    // create a force-directed layout with repulsion, attraction, and collision forces
    const simulation = d3.forceSimulation(imageData)
      .force('collision', d3.forceCollide().radius(collisionDistance))
      .stop();
    // run the simulation for a set number of iterations
    nSimulation = 100;

    // if more than X images make Y steps
    if (imageData.length > 500) {
      nSimulation = 300;
    }

    for (let i = 0; i < nSimulation; i++) {
      simulation.tick(); 
    }

    // centre the post-simulated points so the median is at 0
    const medianX = d3.median(imageData, d => d.x);
    const medianY = d3.median(imageData, d => d.y);
    for (let i = 0; i < imageData.length; i++) {
      imageData[i].x -= medianX;
      imageData[i].y -= medianY;
    }


    // Save the new simulation (x,y) pairs as force_x, force_y

    for (let i = 0; i < imageData.length; i++) {
      imageData[i].force_x = imageData[i].x;
      imageData[i].force_y = imageData[i].y;
    }

    console.log("Range at end " + d3.extent(imageData, d => d.x) + " " + d3.extent(imageData, d => d.y))

    return imageData;
}

async function loadThumbnails(embeddings) {

  console.log("Loading thumbnails...")

  // If dataSource === "online" then the thumbnails are already loaded - just use the urls
  if (dataSource === "online") {
    const imageResults = Object.entries(embeddings).map(([path, embedding]) => {
      return {
        url: path,
        path: path,
        // catalog:catalog
        // metadata[path] contains the catalog - i.e. url to open on click
      };
    });
    return imageResults;
  }


// show progress bar
const progressBarContainer = document.getElementById("renderingProgressBar");
const progressBar = document.getElementById("renderingProgressBarEl");
const timeLeftSpan = document.getElementById("renderingTimeLeft");
progressBarContainer.style.display = "block";

const imageResults = [];

try {
directoryHandleThumb = await directoryHandle.getDirectoryHandle('thumbnails');
} catch {
directoryHandleThumb = await directoryHandle.getDirectoryHandle('thumbnails', { create: true });
}

// Check if thumbnailPath is in directoryHandleThumb
thumbFileNames = [];
for await (let [name,handle] of directoryHandleThumb){
thumbFileNames.push(name);
}

// create an array of promises for loading all thumbnails
const promises = Object.entries(embeddings).map(async ([path, embedding]) => {
let url;
let handle;

// // index 
// const i = Object.keys(embeddings).indexOf(path);
// const score = umap_embedding[i][0];
// const score2 = umap_embedding[i][1];

// if path is not already a URL
if (!path.startsWith("http")) {
// console.log("Getting thumb " , path)
url = await getThumbnail(path, directoryHandleThumb, thumbFileNames);
} else {
url = path;
}

imageResults.push({ url, path });

// update progress bar
progressBar.value += 1;

// // calculate estimated time remaining
// const progress = progressBar.value / progressBar.max;
// const elapsedTime = (Date.now() - startTime) / 1000;
// const estimatedTimeRemaining = (elapsedTime / progress) - elapsedTime;

// // update progress bar label
// const timeLeft = formatTime(estimatedTimeRemaining);
// timeLeftSpan.innerHTML = `Time left: ${timeLeft}`;

});

// initialize progress bar
progressBar.max = promises.length;
progressBar.value = 0;

// record start time
// const startTime = Date.now();

// wait for all promises to resolve
await Promise.all(promises);

// hide progress bar
progressBarContainer.style.display = "none";

return imageResults;
}

function formatTime(seconds) {
const minutes = Math.floor(seconds / 60);
const remainingSeconds = Math.floor(seconds % 60);
return `${minutes < 10 ? '0' : ''}${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
}


function clearD3(){
  d3.select("svg").remove();
}



async function getThumbnail(path, directoryHandleThumb, thumbFileNames, maxSize = 0.2) {
// Roughly 10 kB - thus 0.01 megabytes - for a 256x256 image. Only bother if it is more than 20 times bigger, i.e. more than .2 MB
  const fileName = path.split("/").pop();
const thumbnailPath = `thumb_${fileName}`;  

const thumbnailExists = thumbFileNames.includes(thumbnailPath);
// console.log("thumbnailExists", thumbnailPath);


if ( thumbnailExists) {
// load thumbnail from file system
const thumbnailFile = await getThumbnailFile(directoryHandleThumb,thumbnailPath);
return URL.createObjectURL(thumbnailFile);
}


// check if thumbnail already exists in .thumbnails folder
// const thumbnailPath = `./.thumbnails/${handle.name}`;


const handle = await getFileHandleByPath(path);
const maxDimension = 256;
const blob = await handle.getFile();
const sizeInMB = blob.size / (1024 * 1024);

if (sizeInMB <= maxSize){
// console.log('loaded from file system')
// load image direct from file system
return URL.createObjectURL(blob);
}

// console.log('creating thumbnail')

const image = new Image();
image.src = URL.createObjectURL(blob);
const canvas = document.createElement('canvas');
const context = canvas.getContext('2d');



// change directory handle to 'thumbnails' subfolder

//  console.log("Created/found thumbnails folder")

return new Promise((resolve) => {
image.onload = async () => {
const { width, height } = image;
canvas.width = width;
canvas.height = height;
context.drawImage(image, 0, 0);

let thumbnailWidth = width;
let thumbnailHeight = height;

if (thumbnailWidth > maxDimension) {
  thumbnailWidth = maxDimension;
  thumbnailHeight = Math.round(height * maxDimension / width);
}

if (thumbnailHeight > maxDimension) {
  thumbnailHeight = maxDimension;
  thumbnailWidth = Math.round(width * maxDimension / height);
}

const thumbnailCanvas = document.createElement('canvas');
const thumbnailContext = thumbnailCanvas.getContext('2d');
thumbnailCanvas.width = thumbnailWidth;
thumbnailCanvas.height = thumbnailHeight;
thumbnailContext.drawImage(canvas, 0, 0, width, height, 0, 0, thumbnailWidth, thumbnailHeight);

const thumbnailDataURL = thumbnailCanvas.toDataURL('image/jpeg', 0.8);

// save thumbnail to file system
await saveThumbnailFile(directoryHandleThumb, thumbnailPath, thumbnailCanvas);

resolve(thumbnailDataURL);
};
});
}

async function fileExists(path) {
try {
const file = await  stat(path);
return file.isFile();
} catch {
return false;
}
}

async function getThumbnailFile(dirHand, path) {
const handle = await  dirHand.getFileHandle(path, { create: false });
return await handle.getFile();
}

async function saveThumbnailFile(dirHand, path, canvas) {
// create file and write data to it
const fileHandle = await dirHand.getFileHandle(path, { create: true });
const writable = await fileHandle.createWritable();
canvas.toBlob(async function (blob) {
await writable.write(blob);
await writable.close();
}, 'image/jpeg', 0.8);
}


function sanitizeFilename(filename) {
return filename.replace(/[^a-zA-Z0-9-_.]/g, "_");
}



async function getFileHandleByPath(path) {
  let handle = directoryHandle;
  let chunks = path.split("/").slice(1);
  for(let i = 0; i < chunks.length; i++) {
    let chunk = chunks[i];
    if(i === chunks.length-1) {
      handle = await handle.getFileHandle(chunk);
    } else {
      handle = await handle.getDirectoryHandle(chunk);
    }
  }
  return handle;
}

async function getRgbData(blob) { 
  // let blob = await fetch(imgUrl, {referrer:""}).then(r => r.blob());

  let resizedBlob = await bicubicResizeAndCenterCrop(blob);
  let img = await createImageBitmap(resizedBlob);

  let oscanvas = new OffscreenCanvas(224, 224);
  let ctx = oscanvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  let imageData = ctx.getImageData(0, 0, oscanvas.width, oscanvas.height);

  let rgbData = [[], [], []]; // [r, g, b]
  // remove alpha and put into correct shape:
  let d = imageData.data;
  for(let i = 0; i < d.length; i += 4) { 
    let x = (i/4) % oscanvas.width;
    let y = Math.floor((i/4) / oscanvas.width)
    if(!rgbData[0][y]) rgbData[0][y] = [];
    if(!rgbData[1][y]) rgbData[1][y] = [];
    if(!rgbData[2][y]) rgbData[2][y] = [];
    rgbData[0][y][x] = d[i+0]/255;
    rgbData[1][y][x] = d[i+1]/255;
    rgbData[2][y][x] = d[i+2]/255;
    // From CLIP repo: Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    rgbData[0][y][x] = (rgbData[0][y][x] - 0.48145466) / 0.26862954;
    rgbData[1][y][x] = (rgbData[1][y][x] - 0.4578275) / 0.26130258;
    rgbData[2][y][x] = (rgbData[2][y][x] - 0.40821073) / 0.27577711;
  }
  rgbData = Float32Array.from(rgbData.flat().flat());
  return rgbData;
}

async function bicubicResizeAndCenterCrop(blob) {
  let im1 = vips.Image.newFromBuffer(await blob.arrayBuffer());

  // Resize so smallest side is 224px:
  const scale = 224 / Math.min(im1.height, im1.width);
  let im2 = im1.resize(scale, { kernel: vips.Kernel.cubic });

  // crop to 224x224:
  let left = (im2.width - 224) / 2;
  let top = (im2.height - 224) / 2;
  let im3 = im2.crop(left, top, 224, 224)

  let outBuffer = new Uint8Array(im3.writeToBuffer('.png'));
  im1.delete(), im2.delete(), im3.delete();
  return new Blob([outBuffer], { type: 'image/png' });
}


function downloadBlobWithProgressOld(url, onProgress) {
  return new Promise((res, rej) => {
    var blob;
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.responseType = 'arraybuffer';
    xhr.onload = function(e) {
      blob = new Blob([this.response]);   
    };
    xhr.onprogress = onProgress;
    xhr.onloadend = function(e){
      res(blob);
    }
    xhr.send();
  });
}


function downloadBlobWithProgress(url, onProgress) {
return new Promise((res, rej) => {
const filename = url.substring(url.lastIndexOf('/')+1);
const dbRequest = window.indexedDB.open('myDatabase', 1);
dbRequest.onerror = rej;
dbRequest.onupgradeneeded = function(event) {
const db = event.target.result;
db.createObjectStore('files');
};
dbRequest.onsuccess = function(event) {
const db = event.target.result;
const tx = db.transaction(['files'], 'readonly');
const store = tx.objectStore('files');
const getRequest = store.get(filename);
getRequest.onsuccess = function(event) {
  const fileData = event.target.result;
  if (fileData) {
    // file already exists in IndexedDB, load from dataURL
    const blob = dataURLToBlob(fileData);
    res(blob);
  } else {
    // file does not exist in IndexedDB, download and save
    const xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.responseType = 'blob';
    xhr.onload = function(e) {
      const blob = this.response;
      const reader = new FileReader();
      reader.onloadend = function() {
        const tx = db.transaction(['files'], 'readwrite');
        const store = tx.objectStore('files');
        store.put(reader.result, filename);
      };
      reader.readAsDataURL(blob);
      res(blob);
    };
    xhr.onprogress = onProgress;
    xhr.onerror = rej;
    xhr.send();
  }
};
};
});
}

function dataURLToBlob(dataURL) {
const arr = dataURL.split(',');
const mime = arr[0].match(/:(.*?);/)[1];
const bstr = atob(arr[1]);
let n = bstr.length;
const u8arr = new Uint8Array(n);
while(n--) {
u8arr[n] = bstr.charCodeAt(n);
}
return new Blob([u8arr], {type:mime});
}
// end of IndexedDB code

async function saveEmbeddings(opts={}) {
  let writable = await embeddingsFileHandle.createWritable();
  let textBatch = "";
  let i = 0;
  for(let [filePath, embeddingVec] of Object.entries(embeddings)) {
    let vecString = opts.compress ? JSON.stringify(embeddingVec.map(n => n.toFixed(3))).replace(/"/g, "") : JSON.stringify(embeddingVec);
    textBatch += `${filePath}\t${vecString}\n`;
    i++;
    if(i % 1000 === 0) {
      await writable.write(textBatch);
      textBatch = "";
    }
  }
  await writable.write(textBatch);
  await writable.close();
}

// Tweaked version of example from here: https://developer.mozilla.org/en-US/docs/Web/API/ReadableStreamDefaultReader/read
async function* makeTextFileLineIterator(blob, opts={}) {
  const utf8Decoder = new TextDecoder("utf-8");
  let stream = await blob.stream();
  
  if(opts.decompress === "gzip") stream = stream.pipeThrough(new DecompressionStream("gzip"));
  
  let reader = stream.getReader();
  
  let {value: chunk, done: readerDone} = await reader.read();
  chunk = chunk ? utf8Decoder.decode(chunk, {stream: true}) : "";

  let re = /\r\n|\n|\r/gm;
  let startIndex = 0;

  while (true) {
    let result = re.exec(chunk);
    if (!result) {
      if (readerDone) {
        break;
      }
      let remainder = chunk.substr(startIndex);
      ({value: chunk, done: readerDone} = await reader.read());
      chunk = remainder + (chunk ? utf8Decoder.decode(chunk, {stream: true}) : "");
      startIndex = re.lastIndex = 0;
      continue;
    }
    yield chunk.substring(startIndex, result.index);
    startIndex = re.lastIndex;
  }
  if (startIndex < chunk.length) {
    // last line didn't end in a newline char
    yield chunk.substr(startIndex);
  }
}

function cosineSimilarity(A, B) {
  if(A.length !== B.length) throw new Error("A.length !== B.length");
  let dotProduct = 0, mA = 0, mB = 0;
  for(let i = 0; i < A.length; i++){
    dotProduct += A[i] * B[i];
    mA += A[i] * A[i];
    mB += B[i] * B[i];
  }
  mA = Math.sqrt(mA);
  mB = Math.sqrt(mB);
  let similarity = dotProduct / (mA * mB);
  return similarity;
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function enableCtn(el) {
  el.style.opacity = 1;
  el.style.pointerEvents = "";
}
function disableCtn(el) {
  el.style.opacity = 0.5;
  el.style.pointerEvents = "none";
}


function getCSV() {
// We want 3 columns to our CSV: filename, prompt1 (score), and prompt 2 (score2).
// searchtextel
const prompt1 = document.getElementById('searchTextEl').value;
const prompt2 = document.getElementById('searchTextEl2').value;

// If prompt1 or prompt2 are empty, replace with x-axis or y-axis
prompt1Label = prompt1 || 'x-axis-UMAP';
prompt2Label = prompt2 || 'y-axis-UMAP';

let csv = 'Image,' + prompt1Label + ',' + prompt2Label + '\n';

// Iterate through the d3 data and add a row for each file, with d.score and d.score2 
// as the prompt1 and prompt2 scores.
for (let i = 0; i < imageResults.length; i++) {
const d = imageResults[i];
// Strip any commas from d.path 
const path = d.path.replace(/,/g, '');
csv += path + ',' + d.score + ',' + d.score2 + '\n';
}

// Create a Blob object from the CSV data.
const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });

// Create a URL for the Blob object using the createObjectURL() method.
const url = URL.createObjectURL(blob);

// Create a link element and set its attributes.
const link = document.createElement('a');
link.setAttribute('href', url);
link.setAttribute('download', '2DCLIP.csv');

// Trigger a click event on the link element to download the CSV file.
link.click();
}



{/* </script> */}