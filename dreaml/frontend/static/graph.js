$.getJSON('json/graph',function(data){
  console.log(data)
  var gray = '#555555'
  var cy = cytoscape({
    container: document.getElementById('cy'),
    elements: data,
    style: [
    {
      selector: 'node',
      style: {
        'label':'data(id)',
        'text-opacity': 1,
        'background-color': gray,
        'text-valign': 'bottom',
      'text-outline-color':'#FFFFFF',
      'text-outline-opacity': 0.3,
      'text-outline-width': '2px'
      }
    },
    {
      selector: 'edge',
      style: {
        'curve-style': 'bezier',
        'target-arrow-shape': function(ele) {
          if(ele.data('type')==="explicit"){
            return 'triangle';
          } else {
            return 'none';
          }
        },
        'target-arrow-color': gray,
        'line-style': function(ele) {
          if(ele.data('type')==="explicit"){
            return 'solid';
          } else {
            return 'dashed';
          }
        },
        'line-color': gray,
        'display': function(ele) {
          if(ele.data('display')===true){
            return 'element';
          } else {
            return 'none';
          }
        }
      }
    }],
    layout: {
      name: 'breadthfirst',
      directed: true
    }
  });

  // Mouseover event
  cy.on('mouseover','node',function(evt){
    var node = evt.cyTarget;
    node.style({
      'text-opacity':1,
      'background-color':'#000000',
      'text-outline-color':'#FFFFFF',
      'text-outline-opacity': 0.9,
      'text-outline-width': '2px'
    });
  });

  cy.on('mouseover','edge',function(evt){
    var edge = evt.cyTarget;
    edge.style({
      'line-color':'#000000'
    });
  });

  cy.on('mouseout','node,edge',function(evt){
    var node = evt.cyTarget;
    node.removeStyle();
  });

  // Change layout on select box change
  $("#layout").change(function(){
    var layout = $("#layout").val();
    var options;
    if(layout==="breadthfirstdirected"){
      options = {
        name: "breadthfirst",
        directed: true
      }
    } 
    else {
      options = {
        name: layout
      };
    }
    options.animate = true;
    options.animationDuration = 500;
    options.animationThreshold = 250;
    cy.layout(options);
  });
});