module.exports = function(grunt) {
  'use strict';

  var jshintrc = '.jshintrc';
  var gruntFile = 'Gruntfile.js';
  var directoryPackage = './code';
  var directoryPrivate = directoryPackage + '/private';
  var directoryCodeMirror = directoryPackage + '/codemirror';
  var directoryPublic = directoryPackage + '/public';
  var directoryPrivateJs = directoryPrivate + '/js';
  var directoryPrivateCss = directoryPrivate + '/css';
  var directoryPrivateJsAll = directoryPrivateJs + '/js';
  var directoryPrivateLessAll = directoryPrivate + '/css/**/*.less';
  var directoryPrivateHtmlAll = directoryPrivate + '/html/**/*.html';
  var directoryPublicJs = directoryPublic + '/js';
  var directoryPublicCss = directoryPublic + '/css';
  var directoryPublicCodeMirror = directoryPublic + '/codemirror';
  var directoryPublicCssAll = directoryPublicCss + '/**/*.css';

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
      jsStudent : {
        src : [
          directoryPrivateJs + '/student.js',
        ],
        dest : directoryPublicJs + '/student.js',
      },
      jsStudio : {
        src : [
          directoryPrivateJs + '/studio.js',
        ],
        dest : directoryPublicJs + '/studio.js',
      },
      jsWorkbench : {
        src : [
          directoryPrivateJs + '/workbench.js',
        ],
        dest : directoryPublicJs + '/workbench.js',
      },
      cssCodeMirror : {
        src : [
          directoryCodeMirror + '/codemirror.css',
        ],
        dest : directoryPublicCodeMirror + '/codemirror.css',
      },
      cssStudent : {
        src : [
          directoryPrivateCss + '/student_view.less',
        ],
        dest : directoryPublicCss + '/student_view.less',
      },
      cssStudio : {
        src : [
          directoryPrivateCss + '/studio_view.less',
        ],
        dest : directoryPublicCss + '/studio_view.less',
      },
      cssWorkbench : {
        src : [
          directoryPrivateCss + '/workbench_view.less',
        ],
        dest : directoryPublicCss + '/workbench_view.less',
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
            dest : directoryPublic + '/assets',
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
            cwd : directoryPublicCss,
            src : [
              '*.css',
              '!*.min.css',
            ],
            dest : directoryPrivateCss,
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
              directoryPrivate + '/html/student_view.html',
          'code/public/studio_view.html' :
              directoryPrivate + '/html/studio_view.html',
          'code/public/workbench_view.html' :
              directoryPrivate + '/html/workbench_view.html',
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
      student : {
        options : {
          sourceMap : true,
          sourceMapFilename :
              directoryPublicCss + 'student_view.less.min.css.map',
          outputSourceFiles : true,
          cleancss : true,
          compress : true,
        },
        files : {
          'code/public/css/student_view.less.min.css' :
              directoryPublicCss + '/student_view.less',
        },
      },
      studio : {
        options : {
          sourceMap : true,
          sourceMapFilename :
              directoryPublicCss + 'studio_view.less.min.css.map',
          outputSourceFiles : true,
          cleancss : true,
          compress : true,
        },
        files : {
          'code/public/css/studio_view.less.min.css' :
              directoryPublicCss + '/studio_view.less',
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
            dest : directoryPublicCodeMirror + '/keymap',
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
            dest : directoryPublicCodeMirror + '/modes',
            ext : '.js.min.js',
            flatten : true,
          }
        ],
      },
      scripts : {
        files : [
          {
            expand : true,
            dest : directoryPublicCodeMirror,
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
            cwd : directoryPublicJs + '/',
            src : [
              '*.js',
              '!*.min.js',
            ],
            dest : directoryPublicJs,
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
