let modal = document.createElement('div')
modal.setAttribute('id','modal-window')
modal.setAttribute('class','modal')
modal.innerHTML = ` 
  <span class="close">&times;</span>
  <img class="modal-content" id="modal-image">
  <div id="caption"></div>
` 
document.body.appendChild(modal) 

document.querySelectorAll("img").forEach( (el,idx) => {
  el.setAttribute('title','Klik voor een groter plaatje')
  el.addEventListener ('click', (el) => {
      document.querySelector("#modal-image").src = el.target.currentSrc
      document.querySelector("#modal-image").setAttribute('title','Klik om te verbergen')
      document.querySelector("#caption").innerHTML = el.target.alt
      modal.style.display = "block";
    })
})


modal.addEventListener('click', () => modal.style.display = "none" ) 

document.querySelectorAll('a').forEach ( el => el.addEventListener('click', evt => {
  const t = evt.target.href
  if (t.indexOf('hanze-hbo-ict.github.io') == -1 && t.indexOf('localhost') == -1) {
    evt.target.setAttribute('target', '_blank')
  }
  
})
)
