import logging
import pkg_resources
import json
import cgi
from markdown import markdown
from xblock.core import XBlock
from xblock.fields import Scope, Integer, String, Boolean
from xblock.fragment import Fragment


log = logging.getLogger(__name__)

DeviceQueryWokbenchDescription = """\

## Objective

The purpose of this lab is to get you familiar with using the submission system for this course and the hardware used.

## Instructions

Click on the code tab and then read the code written.
Do not worry if you do not understand all the details of the code (the purpose is to get you familiar with the submission system).
Once done reading, click the "Compile & Run" button.

The submission system will automatically switch to the compile-and-run results that will also be available through the **Attempts** tab.
There, you will be able to see a summary of your attempt.

The `Timer` section has 3 columns:

* *Kind* corresponds with the first argument to `wbTimer_start`,
* *Location* describes the `file::line_number` of the `wbTimer` call, and
* *Time* in millisecond that it took to execute the code in between the `wbTime_start` and `wbTime_stop`, and
* *Message* the string you passed into the second argument to the timer

Similarly, you will see the following information under the `Logger` section.

The `Logger` section has 3 columns:

* *Level* is the level specified when calling the `wbLog` function (indicating the severity of the event),
* *Location* describes the `function::line_number` of the `wbLog` call, and
* *Message* which is the message specified for the `wbLog` function

The `Timer` or `Logger` seconds are hidden, if no timing or logging statements occur in your program.

We log the hardware information used for this course --- the details which will be explained in the first few lectures.

* GPU card's name

* GPU computation capabilities

* Maximum number of block dimensions

* Maximum number of grid dimensions

* Maximum size of GPU memory

* Amount of constant and share memory

* Warp size

All results from previous attempts can be found in the Attempts tab.
You can choose any of these attempts for submission for grading.
Note that even though you can submit multiple times, only your last submission will be reflected in the Coursera database.

After completing this lab, and before proceeding to the next one, you will find it helpful to read the [tutorial](/help) document
"""
DeviceQueryWorkbenchCode = """\
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

class DescriptionXBlock(XBlock):
    description = String(
        help="The lab description",
        default="",
        scope=Scope.content
    )

    def student_view(self, context=None):
        frag = Fragment()
        frag.add_content(markdown(self.description))
        return frag

    @classmethod
    def parse_xml(cls, node, runtime, keys, id_generator):
        block = runtime.construct_xblock_from_class(cls, keys)
        block.description = unicode(node.text or u"")
        return block


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
        if len(self.children) == 0:
            frag.initialize_js('CodeProblemXBlock')
        return frag


    @staticmethod
    def workbench_scenarios():
        """A canned scenario for display in the workbench."""
        return [
            ("CodeXBlock",
             """
                <codeproblem>
                    <description>"""+
                        cgi.escape(DeviceQueryWokbenchDescription).encode('ascii', 'xmlcharrefreplace') +
                    """</description>
                    <template_code>""" +
                        cgi.escape(DeviceQueryWorkbenchCode).encode('ascii', 'xmlcharrefreplace') +
                    """</template_code>
                </codeproblem>
             """
            ),
        ]
