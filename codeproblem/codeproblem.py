import logging
import pkg_resources
import json
import cgi
from xblock.core import XBlock
from xblock.fields import Scope, Integer, String, Boolean
from xblock.fragment import Fragment


log = logging.getLogger(__name__)


DeviceQueryTemplateCode = """\
#include	<wb.h>

//@@ The purpose of this code is to become familiar with the submission
//@@ process. Do not worry if you do not understand all the details of
//@@ the code.

int main(int argc, char ** argv) {
    int deviceCount;

    wbArg_read(argc, argv);

    cudaGetDeviceCount(&deviceCount);

    wbTime_start(GPU, "Getting GPU Data."); //@@ start a timer

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;

        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
                wbLog(TRACE, "No CUDA GPU has been detected");
                return -1;
            } else if (deviceCount == 1) {
                //@@ WbLog is a provided logging API (similar to Log4J).
                //@@ The logging function wbLog takes a level which is either
                //@@ OFF, FATAL, ERROR, WARN, INFO, DEBUG, or TRACE and a
                //@@ message to be printed.
                wbLog(TRACE, "There is 1 device supporting CUDA");
            } else {
                wbLog(TRACE, "There are ", deviceCount, " devices supporting CUDA");
            }
        }

        wbLog(TRACE, "Device ", dev, " name: ", deviceProp.name);
        wbLog(TRACE, " Computational Capabilities: ", deviceProp.major, ".", deviceProp.minor);
        wbLog(TRACE, " Maximum global memory size: ", deviceProp.totalGlobalMem);
        wbLog(TRACE, " Maximum constant memory size: ", deviceProp.totalConstMem);
        wbLog(TRACE, " Maximum shared memory size per block: ", deviceProp.sharedMemPerBlock);
        wbLog(TRACE, " Maximum block dimensions: ", deviceProp.maxThreadsDim[0], " x ",
                                                    deviceProp.maxThreadsDim[1], " x ",
                                                    deviceProp.maxThreadsDim[2]);
        wbLog(TRACE, " Maximum grid dimensions: ", deviceProp.maxGridSize[0], " x ",
                                                   deviceProp.maxGridSize[1], " x ",
                                                   deviceProp.maxGridSize[2]);
        wbLog(TRACE, " Warp size: ", deviceProp.warpSize);
    }

    wbTime_stop(GPU, "Getting GPU Data."); //@@ stop the timer

    return 0;
}
"""


class CodeXBlock(XBlock):
    template_code = String(
        help="The lab template code",
        default="",
        scope=Scope.content
    )
    code = String(
        help="The lab code",
        default="",
        scope=Scope.user_state
    )
    readOnly = Boolean(
        help="Should this code be editable?",
        default=False,
        scope=Scope.content
    )

    def resource_string(self, path):
        """Handy helper for getting resources from our kit."""
        data = pkg_resources.resource_string(__name__, path)
        return data.decode("utf8")

    @XBlock.json_handler
    def code_save(self, submissions, suffix=''):
        if not isinstance(submissions, dict):
            log.error(
                "submissions object from Studio is not a dict - %r", submissions)
            return {
                'response': 'error',
                'message': 'input is not a dictionary',
            }
        try:
            self.code = submissions["code"]
        except KeyError as ex:
            return {
                'response': 'error',
                'message': 'code field was not found',
                'status_code': 400,
            }
        return {'response': 'success'}

    def _add_codemirror_frag(self, frag):
        # frag.add_css(self.resource_string("public/codemirror/codemirror.css"))
        # frag.add_javascript(self.resource_string("public/codemirror/codemirror.js.min.js"))
        # frag.add_javascript(self.resource_string("public/codemirror/modes/clike.js.min.js"))
        frag.add_css_url(
            "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/codemirror.css")
        frag.add_css_url(
            "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/addon/hint/show-hint.css")
        frag.add_css_url(
            "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/theme/eclipse.css")
        frag.add_javascript_url(
            "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/codemirror.js")
        frag.add_javascript_url(
            "https://cdnjs.cloudflare.com/ajax/libs/underscore.js/1.8.3/underscore-min.js")
        frag.add_javascript_url(
            "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/addon/edit/matchbrackets.js")
        frag.add_javascript_url(
            "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/addon/edit/matchbrackets.js")
        frag.add_javascript_url(
            "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/addon/hint/show-hint.js")
        frag.add_javascript_url(
            "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/mode/clike/clike.js")
        # frag.add_javascript(self.resource_string("public/js/cuda-mode.js.min.js"))
        frag.add_javascript(self.resource_string("public/js/cuda-mode.js"))
        return

    def student_view(self, context=None):
        frag = Fragment()
        html = self.resource_string("public/code_view.html")
        self._add_codemirror_frag(frag)
        frag.add_css(self.resource_string(
            "public/css/student_view.less.min.css"))
        if self.code == "":
            self.code = self.template_code
        frag.add_content(html.format(self=self))
        # frag.add_javascript(self.resource_string("public/js/code.js.min.js"))
        frag.add_javascript(self.resource_string("public/js/code.js"))
        frag.initialize_js('CodeXBlock')
        return frag

    def studio_view(self, context=None):
        frag = Fragment()
        html = self.resource_string("public/code_view.html")
        return frag

    @classmethod
    def parse_xml(cls, node, runtime, keys, id_generator):
        block = runtime.construct_xblock_from_class(cls, keys)
        block.template_code = unicode(node.text or u"")
        return block


class CodeProblemXBlock(XBlock):
    has_children = True

    number = Integer(
        help="The lab number (this must be unique for the course)",
        default=0,
        scope=Scope.content
    )
    name = String(
        help="The lab name",
        default="",
        scope=Scope.content
    )

    description = Integer(
        help="The lab description",
        default="",
        scope=Scope.content
    )
    questions = String(
        help="The lab questions",
        default="",
        scope=Scope.content
    )

    def resource_string(self, path):
        """Handy helper for getting resources from our kit."""
        data = pkg_resources.resource_string(__name__, path)
        return data.decode("utf8")


    def student_view(self, context=None):
        """
        A primary view of the CodeXBlock, shown to staff
        when editting the course.
        """
        frag = Fragment()
        frag.add_css(self.resource_string(
                "public/css/studio_view.less.min.css"))
        # frag.add_javascript(self.resource_string("public/js/student.js.min.js"))
        frag.add_javascript(self.resource_string("public/js/student.js"))


        html = self.resource_string("public/student_view.html")
        frag.add_content(html.format(self=self))
        child_frags = self.runtime.render_children(self, context)
        frag.add_frags_resources(child_frags)
        for child_frag in child_frags:
            frag.add_content(child_frag.content)
        frag.initialize_js('CodeProblemXBlock')
        return frag


    @staticmethod
    def workbench_scenarios():
        """A canned scenario for display in the workbench."""
        return [
            ("CodeXBlock",
             """
                <codeproblem>
                    <template_code>""" +
             cgi.escape(DeviceQueryTemplateCode).encode('ascii', 'xmlcharrefreplace') +
             """</template_code>
                </codeproblem>
             """
            ),
        ]
