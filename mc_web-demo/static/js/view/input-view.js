String.prototype.replaceAll = function(search, replacement) {
    var target = this;
    return target.split(search).join(replacement);
};

define(['template/input-template', 'backbone', 'view/story-loader-view'], function(inputTemplate, Backbone, StoryLoaderView) {

   var  InputView = Backbone.View.extend({

        events: {
            'click .btn-answer': '_onAnswerClick',
            'change .story-text': '_onTextChanged',
            'change .question-text': '_onTextChanged',
            'change .choices-text': '_onTextChanged',
        },

        initialize: function() {
            this.setElement("#input_container");
            this.render();
        },

        render: function() {
            var template = _.template( inputTemplate);
            this.$el.html(template);

            this.$story = this.$el.find('.story-text');
            this.$question = this.$el.find('.question-text');
            this.$choices= this.$el.find('.choices-text');

	          this.storyLoaderView = new StoryLoaderView({ model: this.model, el: this.$el.find('.story-load-btn') });
            //.append(this.storyLoaderView.render().$el);
        },

        _onAnswerClick: function () {
            var story = this.$story.val().trim(),
                question = this.$question.val().trim(),
				choices = this.$choices.val().trim(),
                sentences;

			//sentences = story.replaceAll('.','\n');
			//sentences = sentences.replaceAll('?','\n');
			//sentences = sentences.replaceAll('!','\n');
            //sentences = sentences.split('\n');
            sentences = story.split('\n');
			

            this.model.set("story", sentences);
            this.model.set("question", question);
            this.model.set("choices", choices);

            if (!!story && !!question) {
                this.model.getAnswer();
            }

        },

        _onTextChanged: function () {
            this.model.set('correctAnswer', '', {silent: true})
        },

        setValues: function (story, question, choices) {
            this.$story.val(story);
            this.$question.val(question);
            this.$choices.val(choices);
        }

    });

    return InputView;
});
