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
  for (var i=1; i<indices.length; i++){
    var next_partition = index[indices[i]];
    // If its a different partition then add it
    if(last_partition !== next_partition){
      var prev = compressed[compressed.length-1]
      if (cur_partition_size > 2) {
        var key = prefix(prev)+"...";
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
    var key = prefix(prev)+"...";
    compressed.push(key);
    index[key] = last_partition;
  }
  if(indices[indices.length-1] !== compressed[compressed.length-1]){
    compressed.push(indices[indices.length-1]);
  }
  return compressed;
}

var cur_data;

draw = function(err, data){ 

  var margin = {top: 100, right: 0, bottom: 0, left: 100},
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

    var n = row_labels.length,
        m = col_labels.length;

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
        .attr("x", -6)
        .attr("y", y.rangeBand() / 2)
        .attr("dy", ".32em")
        .attr("text-anchor", "end")
        .text(function(d, i) { return row_labels[i]; });

    var column = svg.selectAll(".column")
        .data(matrix[0])
      .enter().append("g")
        .attr("class", "column")
        .attr("transform", function(d, i) { return "translate(" + x(i) +
          ")rotate(-90)"; });

    column.append("line")
        .attr("x1", -width);

    column.append("text")
        .attr("x", 6)
        .attr("y", x.rangeBand() / 2)
        .attr("dy", ".32em")
        .attr("text-anchor", "start")
        .text(function(d, i) { return col_labels[i]; });

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