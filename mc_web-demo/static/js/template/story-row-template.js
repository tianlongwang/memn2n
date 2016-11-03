define([], function() {

    return '<div class="col-sm-6 col-md-4">' +
            '<div class="thumbnail">' +
              '<div class="caption">' +
                '<p><a href="#" class="btn btn-primary" role="button">Select</a></p>' +
                '<h5>Story</h5>' +
                '<div class="well"><%= story %></div>' +
                '<h5>Question: <%= question %></h5>' +
                '<h5>Choices: <%= choices %></h5>' +
                '<h5>Correct Answer: <%= answer%></h5>' +
              '</div>' +
            '</div>' +
          '</div>'
});


