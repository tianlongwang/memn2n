define([], function() {

    return '<div class="col-md-3">' +
        '<div class="form-group jumbotron">' +
            '<h3>Story</h3>' +
            '<textarea rows="10" class="story-text form-control" placeholder="Story" />' +
            '<h3>Question</h3>' +
            '<input type="text" class="question-text form-control" placeholder="Question" />' +
            '<h3>AnswerA</h3>' +
            '<input type="text" class="answerA-text form-control" placeholder="answerA" />' +
            '<h3>AnswerB</h3>' +
            '<input type="text" class="answerB-text form-control" placeholder="answerB" />' +
            '<h3>AnswerC</h3>' +
            '<input type="text" class="answerC-text form-control" placeholder="answerC" />' +
            '<br/>'+
            '<button class="btn btn-primary btn-answer">Answer</button>' +
            '<div class="story-load-btn">Answer</div>' +
        '</div>' +
    '</div>';
});
