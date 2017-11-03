# CodeXBlock

[![Greenkeeper badge](https://badges.greenkeeper.io/abduld/xblock-code.svg)](https://greenkeeper.io/)
Source code text area xblock

## TODO List:
- [ ] Write tests
- [ ] Update the `student_view`
    - [ ] `./code/private/view.html`
        - Add content to `<div class="code_block"></div>` element
    - [ ] `./code/private/view.js`
        - Add logic to `CodeView` function
    - [ ] `./code/private/view.less`
        - Add styles to `.code_block { }` block
    - [ ] `./code/code.py`
        - Add back-end logic to `student_view` method
- [ ] Update the `studio_view`
    - [ ] `./code/private/edit.html`
        - Add `<LI>` entries to `<ul class="list-input settings-list">` for each new field
    - [ ] `./code/private/edit.js`
        - Add entry for each field to `CodeEdit`
    - [ ] `./code/private/edit.less`
        - Add styles to `.code_edit { }` block (if needed)
    - [ ] `./code/code.py`
        - Add entry for each field to `studio_view_save`
- [ ] Update package metadata
    - [ ] `./package.json`
        - https://www.npmjs.org/doc/files/package.json.html
    - [ ] `./setup.py`
        - https://docs.python.org/2/distutils/setupscript.html#additional-meta-data
- [ ] Update `./Gruntfile.js`
    - http://gruntjs.com/getting-started
- [ ] Update `./README.markdown`
- [ ] Write documentation
- [ ] Publish on PyPi
