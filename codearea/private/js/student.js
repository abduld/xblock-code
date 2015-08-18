/* Javascript for CodeXBlock. */
function CodeXBlock(runtime, element, data) {

  var readOnly = data.read_only | false;

  $(function($) {
    var editor, save_code;

    save_code = function() {

      var handlerUrl = runtime.handlerUrl(element, 'code_save');
      $.post(handlerUrl, JSON.stringify({
        "code": editor.getValue()
      })).done(function(response) {
        if (response.result === 'success') {
          console.log('Error: ' + response.message);
        }
      });
    };

    editor = CodeMirror.fromTextArea($('.code-editor')[0], {
      lineNumbers: true,
      smartIndent: true,
      tabSize: 4,
      indentUnit: 4,
      indentWithTabs: true,
      readOnly: readOnly,
      theme: 'eclipse',
      mode: 'text/x-cuda-src',
      matchBrackets: true,
      extraKeys: {
        'Ctrl-Space': 'autocomplete'
      }
    });

    window.editor = editor;
    editor.setSize(null, '80%');

    $('.save-code').click(save_code);

  });
}
