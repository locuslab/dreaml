function is_directory(obj){
  return obj["directory"];
}

function get_query(obj){
  return obj["query"];
}

function get_files(obj){
  return obj["files"];
}

function compare_file(a,b){
  if (a.query < b.query) return -1;
  if (a.query > b.query) return 1;
  return 0;
}

function indices_from_json(arr){
  var indices = [];
  var file_list = [];
  for (i in arr) {
  var obj = arr[i];
    if (is_directory(obj)){
      var prefix = get_query(obj);
      var files = (get_files(obj))

      var subindices = indices_from_json(get_files(obj));
      indices.push(prefix);
      indices = indices.concat(subindices.map(function(s){
                  return prefix.concat(s)
                }));
    } else {
      indices.push(get_query(obj));
      // file_list.push(get_query(obj));
    }
  }
  // if (file_list.length > 1) {
  //   file_list.sort();
  //   var first = file_list[0];
  //   var last = file_list[file_list.length-1];
  //   indices = indices.concat([first,last]);

  // } else if (file_list.length == 1){
  //   console.log(file_list)
  //   indices = indices.concat(file_list);
  // }
  return indices;
}

function prefix(path){
  return path.substring(0,path.lastIndexOf("/")+1);
}

function compress_indices(indices,index){
  var compressed = [];
  compressed.push(indices[0]);
  var last_partition = index[indices[0]];
  var cur_partition_size = 1;
  var dots = String.fromCharCode(183)+String.fromCharCode(183)+String.fromCharCode(183);
  for (var i=1; i<indices.length; i++){
    var next_partition = index[indices[i]];
    // If its a different partition then add it
    if(last_partition !== next_partition){
      var prev = compressed[compressed.length-1]
      if (cur_partition_size > 2) {
        var key = prefix(prev)+dots;
        compressed.push(key);
        index[key] = last_partition;
      }
      if(indices[i-1] !== compressed[compressed.length-1]){
        compressed.push(indices[i-1]); 
      }
      compressed.push(indices[i]);
      last_partition = next_partition;
      cur_partition_size = 1;
    } else {
      cur_partition_size++;
    }
  }
  if (cur_partition_size > 2) {
    var prev = compressed[compressed.length-1]
    var key = prefix(prev)+dots;
    compressed.push(key);
    index[key] = last_partition;
  }
  if(indices[indices.length-1] !== compressed[compressed.length-1]){
    compressed.push(indices[indices.length-1]);
  }
  return compressed;
}

var open_directories;

function is_open(obj){

}

function parse_structure(structure,dir){
  var list = [];
  structure.forEach(function(obj,i){
        if(is_directory(obj) && is_open(obj)){
            var query = dir.concat("/"+get_query(obj));
            list = list.concat(parse_structure(get_files(obj),query));
        } else {
            list.push(get_query(obj));
        }
  });
  return list;
}

function get_label_function(labels){
    return function(d,i){
        return labels[i];
    }
}

function get_label_end_function(labels){
    return function(d,i){
        if(labels[i].endsWith('/')){
            return labels[i].split('/').reverse()[1]+'/';
        } else{
            return labels[i].split('/').pop(); 
        }

    }
}

function get_label_depth_function(labels,scale){
    return function(d,i){
        var depth = labels[i].split('/').length;;
        if(labels[i].endsWith('/')){
            depth--;
        }
        return depth*scale;
    }
}

draw = function(err, data){ 
  console.log(parse_structure(data.rows));

  var margin = {top: 250, right:50, bottom: 0, left: 150},
      width = $("#grid").width()-margin.left,
      height = $("#grid").height()-margin.top;

  var x = d3.scale.ordinal().rangeBands([0,width]),
      y = d3.scale.ordinal().rangeBands([0,height]),
      c = d3.scale.category20().domain(d3.range(20));

  var svg = d3.select("#grid").append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var matrix = [],
        rows = data.rows,
        cols = data.cols;

    var row_index = data.row_index,
        col_index = data.col_index,
        partitions = data.partitions;

    var row_labels = compress_indices(indices_from_json(rows),row_index);
    var col_labels = compress_indices(indices_from_json(cols),col_index);
    // var row_labels = parse_structure(data.rows);
    // var col_labels = parse_structure(data.cols);

    var n = row_labels.length,
        m = col_labels.length;

    // Spacing for levels in the hierarchical view
    var spacing = 15;

    // Compute index per entry.
    row_labels.forEach(function(row, i) {
      matrix[i] = d3.range(m).map(function(j){ return {x: j, y: i, z: 0}; });
    });

    // Convert links to matrix; count character occurrences.
    row_labels.forEach(function(row, i) {
      col_labels.forEach(function(col,j){
        var r_idx = row_index[row];
        var c_idx = col_index[col];
        var pair = "("+r_idx+", "+c_idx+")";
        if (pair in partitions){
          var p_id = partitions[pair]; 
        } else {
          var p_id = -1;
        }
        matrix[i][j].z = p_id;
        matrix[i][j].r = r_idx;
        matrix[i][j].c = c_idx;

        matrix[i][j].row_name = row;
        matrix[i][j].col_name = col;
      });
    });
    
    // default range
    x.domain(d3.range(m));
    y.domain(d3.range(n));

    svg.append("rect")
        .attr("class", "background")
        .attr("width", width)
        .attr("height", height);

    var row = svg.selectAll(".row")
        .data(matrix)
        .enter().append("g")
        .attr("class", "row")
        .attr("transform", function(d, i) {  return "translate(0," + y(i) + ")";
      })
        .each(row);

    row.append("line")
        .attr("x2", width);

    row.append("text")
        .attr("depth", get_label_depth_function(row_labels,1))
        .attr("x", get_label_depth_function(row_labels,spacing))
        .attr("y", y.rangeBand() / 2)
        .attr("dy", ".32em")
        .attr("id",get_label_function(row_labels))
        // .attr("text-anchor", "end")
        .text(get_label_end_function(row_labels))
        .style("font-size","34px");

    var column = svg.selectAll(".column")
        .data(matrix[0])
      .enter().append("g")
        .attr("class", "column")
        .attr("transform", function(d, i) { return "translate(" + x(i) +
          ")rotate(-90)"; });

    column.append("line")
        .attr("x1", -width);

    column.append("text")
        .attr("depth", get_label_depth_function(col_labels,1))
        .attr("x", get_label_depth_function(col_labels,spacing))
        .attr("y", x.rangeBand() / 2)
        .attr("dy", ".32em")
        .attr("text-anchor", "start")
        .text(get_label_end_function(col_labels))
        .style("font-size","34px");
      //   .attr("transform", function(d, i) { return "rotate(90)translate(-10)rotate(-90)translate (27)rotate (45)";
      // });

    function row(row) {
      var cell = d3.select(this).selectAll(".cell")
          .data(row.filter(function(d) { return d.z >= 0; }))
        .enter().append("rect")
          .attr("class", "cell")
          .attr("x", function(d) { return x(d.x); })
          .attr("width", x.rangeBand())
          .attr("height", y.rangeBand())
          // .style("fill-opacity", function(d) { return z(d.z); })
          .style("fill", function(d) { 
            return c(d.z% 20); 
          })
          .on("mouseover", mouseover)
          .on("mouseout", mouseout);
    }

    function mouseover(p) {
      d3.selectAll(".row text").classed("active", function(d, i) { return i == p.y; });
      d3.selectAll(".column text").classed("active", function(d, i) { return i == p.x; });
    }

    function mouseout() {
      d3.selectAll("text").classed("active", false);
    }

    // This function computes start/stops for each hierachical line needed
    function make_lines(text_coords, ordinal){
      var path = [];
      for(var i=0; i<text_coords.length; i++){
        rt = text_coords[i];
        end = text_coords.length;
        for(var j=i+1; j<text_coords.length; j++){
          if (text_coords[j]["depth"] <= rt["depth"]){
            end = j;
            break;
          }
        }
        if (end-1>i){
          var startp = ordinal(i) + ordinal.rangeBand()/2;
          if(end>=text_coords.length){
            var endp = ordinal(end-1) + ordinal.rangeBand();
          } else {
            var endp = ordinal(end);
          }
          path.push([
            {"x":rt["x"],"y":startp},
            {"x":rt["x"],"y":endp}]);
        }
      }
      return path;
    }

    // Extract x,y,depth information from the text svg elements
    var row_text_coords = d3.selectAll("g .row text")[0].map(function(d){
      return {
        "x":d.attributes.x.value,
        "y":d.attributes.y.value,
        "depth":d.attributes.depth.value
      }
    });
    // Calculate the max width of an svg element to translate it left
    var widths = (d3.selectAll("g .row text")[0].map(function(d){
          return d.getBBox().width +(1+parseInt(d.attributes.depth.value))*spacing;
        }));
    var max_width = (Math.max(...widths));
    var row_path = make_lines(row_text_coords,y);
    for(i in row_path){
      row_path[i][0]["x"]-=max_width;
      row_path[i][1]["x"]-=max_width;
    }

    d3.selectAll("g .row text").attr("x",function(d,i){
        return get_label_depth_function(row_labels,spacing)(d,i)-max_width
    });


    var column_text_coords = d3.selectAll("g .column text")[0].map(function(d){
      return {
        "x":d.attributes.x.value,
        "y":d.attributes.y.value,
        "depth":d.attributes.depth.value
      }
    });
    var column_path = make_lines(column_text_coords,x);

    var lineFunction = d3.svg.line()
                       .x(function(d) {return d["x"]})
                       .y(function(d) {return d["y"]})
                       .interpolate("linear");

    for(var i in row_path){
      var path = row_path[i]
      svg.append("path")
         .attr("d",lineFunction(path))
         .attr("stroke","gray")
         .attr("stroke-width",1)
         .attr("fill","none");
    }
    for(var i in column_path){
      var path = column_path[i]
      svg.append("path")
         .attr("d",lineFunction(path))
         .attr("stroke","gray")
         .attr("stroke-width",1)
         .attr("fill","none").attr("transform", function(d, i) { 
            return "translate(" + x(i) + ")rotate(-90)"; 
          });
    }

};

console.log("first")
d3.json("json/structure", draw)

window.setInterval(function(){
  console.log("running!");
  d3.json("json/structure", function(err,data){
    draw(err,data);
    $("#grid").find("svg:first-child").remove();
  });    
}, 3000);