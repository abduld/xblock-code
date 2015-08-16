import logging
import pkg_resources
import json
from xblock.core import XBlock
from xblock.fields import Scope, String, Boolean
from xblock.fragment import Fragment


log = logging.getLogger(__name__)


class CodeXBlock(XBlock):
    code = String(
        help="The field stores user code",
        default="""
int main(void) {
    return 0;
}
        """,
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
        frag.add_css_url("https://cdnjs.cloudflare.com/ajax/libs/underscore.js/1.8.3/underscore-min.js")
        # frag.add_css(self.resource_string("public/codemirror/codemirror.css"))
        # frag.add_javascript(self.resource_string("public/codemirror/codemirror.js.min.js"))
        #frag.add_javascript(self.resource_string("public/codemirror/modes/clike.js.min.js"))
        frag.add_css_url("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/codemirror.css")
        frag.add_javascript_url("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/codemirror.js")
        frag.add_javascript_url("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.5.0/mode/clike/clike.js")
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
