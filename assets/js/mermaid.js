// Mermaid initialization script
document.addEventListener('DOMContentLoaded', function() {
  // First, add the Mermaid CSS to ensure proper styling
  var mermaidCSS = document.createElement('link');
  mermaidCSS.rel = 'stylesheet';
  mermaidCSS.href = 'https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.css';
  document.head.appendChild(mermaidCSS);
  
  // Load Mermaid script
  var script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js';
  script.onload = function() {
    // Initialize Mermaid with advanced configuration
    mermaid.initialize({
      startOnLoad: false, // We'll manually render
      theme: 'default',
      securityLevel: 'loose',
      flowchart: { useMaxWidth: false, htmlLabels: true },
      er: { useMaxWidth: false },
      sequence: { useMaxWidth: false },
      journey: { useMaxWidth: false },
      gantt: { useMaxWidth: false },
      pie: { useMaxWidth: false },
      requirement: { useMaxWidth: false }
    });
    
    // Find all pre code blocks with class 'language-mermaid'
    var mermaidBlocks = document.querySelectorAll('pre code.language-mermaid');
    
    // Process each mermaid code block
    mermaidBlocks.forEach(function(el, index) {
      // Create a unique ID for this diagram
      var id = 'mermaid-diagram-' + index;
      
      // Create a div for mermaid with the unique ID
      var div = document.createElement('div');
      div.className = 'mermaid';
      div.id = id;
      div.innerHTML = el.textContent;
      
      // Replace the pre element with the mermaid div
      var pre = el.parentElement;
      pre.parentElement.replaceChild(div, pre);
    });
    
    // Now render all diagrams
    mermaid.run();
    
    // Add a backup method in case the above doesn't work
    setTimeout(function() {
      // Check if any diagrams were rendered
      var renderedDiagrams = document.querySelectorAll('.mermaid svg');
      if (renderedDiagrams.length === 0) {
        console.log('Mermaid diagrams not rendered, trying alternative method...');
        
        // Try alternative rendering method
        document.querySelectorAll('.mermaid').forEach(function(el) {
          try {
            mermaid.render('mermaid-svg-' + Math.random().toString(36).substr(2, 9), el.textContent)
              .then(function(result) {
                el.innerHTML = result.svg;
              });
          } catch (error) {
            console.error('Error rendering mermaid diagram:', error);
          }
        });
      }
    }, 1000);
  };
  
  // Add error handling
  script.onerror = function() {
    console.error('Failed to load Mermaid script');
  };
  
  document.head.appendChild(script);
});