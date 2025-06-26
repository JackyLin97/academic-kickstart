// Load Mermaid library
document.addEventListener('DOMContentLoaded', function() {
  // Load Mermaid script
  var script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js';
  script.onload = function() {
    // Initialize Mermaid
    mermaid.initialize({
      startOnLoad: true,
      theme: 'default',
      securityLevel: 'loose',
      flowchart: {
        useMaxWidth: false,
        htmlLabels: true
      }
    });
    
    // Find all pre code blocks with class 'language-mermaid' and render them
    document.querySelectorAll('pre code.language-mermaid').forEach(function(el) {
      // Create a div for mermaid
      var div = document.createElement('div');
      div.className = 'mermaid';
      div.innerHTML = el.textContent;
      
      // Replace the pre element with the mermaid div
      var pre = el.parentElement;
      pre.parentElement.replaceChild(div, pre);
      
      // Trigger rendering
      mermaid.init(undefined, div);
    });
  };
  document.head.appendChild(script);
});