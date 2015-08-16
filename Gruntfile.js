module.exports = function(grunt) {
  'use strict';

  var jshintrc = '.jshintrc';
  var gruntFile = 'Gruntfile.js';
  var directoryPackage = './code';
  var directoryPrivate = directoryPackage + '/private';
  var directoryCodeMirror = directoryPackage + '/codemirror';
  var directoryPublic = directoryPackage + '/public';
  var directoryPrivateJsAll = directoryPrivate + '/**/*.js';
  var directoryPrivateLessAll = directoryPrivate + '/**/*.less';
  var directoryPrivateHtmlAll = directoryPrivate + '/**/*.html';
  var directoryPublicCssAll = directoryPublic + '/**/*.css';

  grunt.initConfig({
    pkg : grunt.file.readJSON('package.json'),
    clean : [
      'node_modules/',
      '**/*.pyc',
    ],
    concat : {
      options : {
        separator : ';\n',
      },
      jsView : {
        src : [
          directoryPrivate + '/view.js',
        ],
        dest : directoryPublic + '/view.js',
      },
      jsEdit : {
        src : [
          directoryPrivate + '/edit.js',
        ],
        dest : directoryPublic + '/edit.js',
      },
      cssCodeMirror : {
        src : [
          directoryCodeMirror + '/codemirror.css',
        ],
        dest : directoryPublic + '/codemirror/codemirror.css',
      },
      cssView : {
        src : [
          directoryPrivate + '/view.less',
        ],
        dest : directoryPublic + '/view.less',
      },
      cssEdit : {
        src : [
          directoryPrivate + '/edit.less',
        ],
        dest : directoryPublic + '/edit.less',
      },
    },
    copy : {
      images : {
        files : [
          {
            expand : true,
            src : [
              directoryPrivate + '/**/*.jpg',
              directoryPrivate + '/**/*.png',
              directoryPrivate + '/**/*.gif',
            ],
            dest : directoryPublic + '/',
          },
        ],
      },
    },
    csslint : {
      dist : {
        src : [
          directoryPublicCssAll,
        ],
      },
    },
    cssmin : {
      combine : {
        files : [
          {
            footer : '\n',
            expand : true,
            cwd : directoryPublic,
            src : [
              '*.css',
              '!*.min.css',
            ],
            dest : directoryPublic,
            ext : '.min.css',
          }
        ],
      },
    },
    htmlmin : {
      all : {
        options : {
          removeComments : true,
          removeCommentsFromCDATA : true,
          collapseWhitespace : true,
          collapseBooleanAttributes : true,
          removeRedundantAttributes : true,
          removeEmptyAttributes : true,
        },
        files : {
          'code/public/student_view.html' :
              directoryPrivate + '/student_view.html',
          'code/public/studio_view.html' :
              directoryPrivate + '/studio_view.html',
        },
      },
    },
    jshint : {
      options : {
        ignores : [ "**/mode/**", "**/addon/**", "**/codemirror.js" ],
      },
      dist : [
        gruntFile,
        directoryPrivateJsAll,
      ],
    },
    less : {
      view : {
        options : {
          sourceMap : true,
          sourceMapFilename : 'code/public/view.less.min.css.map',
          outputSourceFiles : true,
          cleancss : true,
          compress : true,
        },
        files : {
          'code/public/view.less.min.css' : directoryPublic + '/view.less',
        },
      },
      edit : {
        options : {
          sourceMap : true,
          sourceMapFilename : 'code/public/student_view.less.min.css.map',
          outputSourceFiles : true,
          cleancss : true,
          compress : true,
        },
        files : {
          'code/public/student_view.less.min.css' :
              directoryPublic + '/student_view.less',
        },
      },
    },
    uglify : {
      options : {
        footer : '\n',
        sourceMap : true,
      },
      keymaps : {
        files : [
          {
            expand : true,
            cwd : directoryCodeMirror + '/keymap',
            src : '*.js',
            dest : directoryPublic + '/codemirror/keymap',
            ext : '.js.min.js',
          }
        ],
      },
      modes : {
        files : [
          {
            expand : true,
            cwd : directoryCodeMirror + '/mode/',
            src : [ '**/*.js' ],
            dest : directoryPublic + '/codemirror/modes',
            ext : '.js.min.js',
            flatten : true,
          }
        ],
      },
      scripts : {
        files : [
          {
            expand : true,
            dest : directoryPublic + '/codemirror',
            cwd : directoryCodeMirror + '/',
            src : [
              'codemirror.js',
              'addon/dialog/dialog.js',
              'addon/edit/closebrackets.js',
              'addon/edit/matchbrackets.js',
              'addon/mode/loadmode.js',
              'addon/mode/overlay.js',
              'addon/mode/simple.js',
              'addon/search/matchesonscrollbar.js',
              'addon/search/searchcursor.js',
              'addon/search/search.js',
              'addon/selection/active-line.js',
              'addon/scroll/annotatescrollbar.js',
              'addon/scroll/simplescrollbars.js',
            ],
            ext : 'js.min.js'
          }
        ]
      },
      combine : {
        files : [
          {
            expand : true,
            cwd : directoryPublic + '/',
            src : [
              '*.js',
              '!*.min.js',
            ],
            dest : directoryPublic + '/',
            ext : '.js.min.js',
          }
        ],
      },
    },
    watch : {
      dist : {
        files : [
          jshintrc,
          gruntFile,
          directoryPrivateJsAll,
          directoryPrivateLessAll,
          directoryPrivateHtmlAll,
        ],
        tasks : [
          'default',
        ],
      },
    },
  });

  grunt.loadNpmTasks('grunt-contrib-jshint');
  grunt.loadNpmTasks('grunt-contrib-csslint');
  grunt.loadNpmTasks('grunt-contrib-uglify');
  grunt.loadNpmTasks('grunt-contrib-cssmin');
  grunt.loadNpmTasks('grunt-contrib-concat');
  grunt.loadNpmTasks('grunt-contrib-less');
  grunt.loadNpmTasks('grunt-contrib-watch');
  grunt.loadNpmTasks('grunt-contrib-copy');
  grunt.loadNpmTasks('grunt-contrib-clean');
  grunt.loadNpmTasks('grunt-contrib-htmlmin');

  grunt.registerTask('default', [
    'jshint',
    'concat',
    'copy',
    'less',
    'csslint',
    'uglify:modes',
    'uglify:scripts',
    'uglify:keymaps',
    'uglify',
    'htmlmin',
  ]);
};
