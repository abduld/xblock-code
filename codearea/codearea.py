import logging
import pkg_resources
import json
from xblock.core import XBlock
from xblock.fields import Scope, String, Boolean
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
    code = String(
        help="The field stores user code",
        default=DeviceQueryTemplateCode,
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

    def _add_codemirror_frag(self, frag):
        # frag.add_css(self.resource_string("public/codemirror/codemirror.css"))
        # frag.add_javascript(self.resource_string("public/codemirror/codemirror.js.min.js"))
        #frag.add_javascript(self.resource_string("public/codemirror/modes/clike.js.min.js"))
        frag.add_css_url("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/codemirror.css")
        frag.add_css_url("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/addon/hint/show-hint.css")
        frag.add_css_url("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/theme/eclipse.css")
        frag.add_javascript_url("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/codemirror.js")
        frag.add_javascript_url("https://cdnjs.cloudflare.com/ajax/libs/underscore.js/1.8.3/underscore-min.js")
        frag.add_javascript_url("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/addon/edit/matchbrackets.js")
        frag.add_javascript_url("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/addon/edit/matchbrackets.js")
        frag.add_javascript_url("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/addon/hint/show-hint.js")
        frag.add_javascript_url("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/mode/clike/clike.js")
        frag.add_javascript(self.resource_string("public/js/cuda-mode.js"))
        return

    @XBlock.json_handler
    def code_save(self, submissions, suffix=''):
        if not isinstance(submissions, dict):
            log.error("submissions object from Studio is not a dict - %r", submissions)
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

    def student_view(self, context=None):
        """
        A primary view of the CodeXBlock, shown to students
        when viewing courses.
        """
        html = self.resource_string("public/student_view.html")
        frag = Fragment(html.format(self=self))
        self._add_codemirror_frag(frag)
        frag.add_css(self.resource_string("public/css/student_view.less.min.css"))
        # frag.add_javascript(self.resource_string("public/js/code.js.min.js"))
        # frag.add_javascript(self.resource_string("public/js/student.js.min.js"))
        frag.add_javascript(self.resource_string("public/js/code.js"))
        frag.add_javascript(self.resource_string("public/js/student.js"))
        frag.initialize_js('CodeXBlock', {
            "read_only": self.readOnly
        })
        return frag

    def studio_view(self, context=None):
        """
        A primary view of the CodeXBlock, shown to staff
        when creating courses.
        """
        html = self.resource_string("public/studio_view.html")
        frag = Fragment(html.format(self=self))
        self._add_codemirror_frag(frag)
        frag.add_css(self.resource_string("public/css/studio_view.less.min.css"))
        frag.add_javascript(self.resource_string("public/js/code.js.min.js"))
        frag.add_javascript(self.resource_string("public/js/studio.js.min.js"))
        frag.initialize_js('CodeXBlock', {
            "read_only": self.readOnly
        })
        return frag


    @staticmethod
    def workbench_scenarios():
        """A canned scenario for display in the workbench."""
        return [
            ("CodeXBlock",
             """
             <vertical_demo>
             <codearea/>
             </vertical_demo>
             """),
        ]
