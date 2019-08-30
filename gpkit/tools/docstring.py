"Docstring-parsing methods"
import re
import inspect
import ast
import numpy as np


class parse_variables(object):

    def __init__(self, string, globals):
        self.string = string
        self.globals = globals

    def __call__(self, f):
        string = self.string
        orig_lines, lineno = inspect.getsourcelines(f)
        indent_length = 0
        while orig_lines[1][indent_length] in [" ", "\t"]:
            indent_length += 1
        first_indent_length = indent_length
        next_indented_idx = 2
        # get the next indented line
        while len(orig_lines[next_indented_idx]) <= indent_length + 1:
            next_indented_idx += 1
        while orig_lines[next_indented_idx][indent_length] in [" ", "\t"]:
            indent_length += 1
        second_indent = orig_lines[next_indented_idx][:indent_length]
        parse_lines = [second_indent + line + "\n"
                       for line in parse_varstring(string, errorcatch=False).split("\n")]
        # make ast of these new lines, insert it into the original ast
        new_lines = [orig_lines[1]] + parse_lines + orig_lines[2:]
        # print(new_lines)
        new_src = "\n".join([line[first_indent_length:-1] for line in new_lines])
        # print("ns\n%s" % new_src)
        # print(lineno, new_src)
        new_ast = ast.parse(new_src, "<parse_variables>")
        # ast.fix_missing_locations(new_ast)
        # print(len(parse_lines), parse_lines)
        ast.increment_lineno(new_ast, n=lineno-len(parse_lines))
        code = compile(new_ast, inspect.getsourcefile(f), "exec", dont_inherit=True)
        out = {}
        exec(code, self.globals, out)
        return out[f.__name__]


def expected_unbounded(instance, doc):
    "Gets expected-unbounded variables from a string"
    # pylint: disable=too-many-locals,too-many-nested-blocks
    exp_unbounded = set()
    for direction in ["upper", "lower"]:
        flag = direction[0].upper()+direction[1:]+" Unbounded\n"
        count = doc.count(flag)
        if count == 0:
            continue
        elif count > 1:
            raise ValueError("multiple instances of %s" % flag)
        idx = doc.index(flag) + len(flag)
        idx2 = doc[idx:].index("\n")
        try:
            idx3 = doc[idx:][idx2+1:].index("\n\n")
        except ValueError:
            idx3 = doc[idx:][idx2+1:].index("\n")
        varstrs = doc[idx:][idx2+1:][:idx3].strip()
        varstrs = varstrs.replace("\n", ", ")  # cross newlines
        varstrs = re.sub(" +", " ", varstrs)   # multiple-whitespace removal
        if varstrs:
            for var in varstrs.split(", "):
                if " (if " in var:  # it's a conditional!
                    var, condition = var.split(" (if ")
                    assert condition[-1] == ")"
                    condition = condition[:-1]
                    invert = condition[:4] == "not "
                    if invert:
                        condition = condition[4:]
                        if getattr(instance, condition):
                            continue
                    elif not getattr(instance, condition):
                        continue
                try:
                    obj = instance
                    for subdot in var.split("."):
                        obj = getattr(obj, subdot)
                    variables = obj
                except AttributeError:
                    raise AttributeError("`%s` is noted in %s as "
                                         "unbounded, but is not "
                                         "an attribute of that model."
                                         % (var, instance.__class__.__name__))
                if not hasattr(variables, "shape"):
                    variables = np.array([variables])
                it = np.nditer(variables, flags=['multi_index', 'refs_ok'])
                while not it.finished:
                    i = it.multi_index
                    it.iternext()
                    exp_unbounded.add((variables[i].key, direction))
    return exp_unbounded


def parse_varstring(string, errorcatch=True):
    "Parses a string to determine what variables to create from it"
    out = "from gpkit import Variable, VectorVariable\n"
    out += check_and_parse_flag(string, "Constants\n", errorcatch,
                                constant_declare)
    out += check_and_parse_flag(string, "Variables\n", errorcatch)
    out += check_and_parse_flag(string, "Variables of length", errorcatch,
                                vv_declare)
    return out


def vv_declare(string, flag, idx2, countstr):
    "Turns Variable declarations into VectorVariable ones"
    length = string[len(flag):idx2].strip()
    return countstr.replace("Variable(", "VectorVariable(%s, " % length)


# pylint: disable=unused-argument
def constant_declare(string, flag, idx2, countstr):
    "Turns Variable declarations into Constant ones"
    return countstr.replace("')", "', constant=True)")


def check_and_parse_flag(string, flag, errorcatch, declaration_func=None):
    "Checks for instances of flag in string and parses them."
    overallstr = ""
    for _ in range(string.count(flag)):
        countstr = ""
        idx = string.index(flag)
        string = string[idx:]
        if idx == -1:
            idx = 0
            skiplines = 0
        else:
            skiplines = 2
        idx2 = 0 if "\n" in flag else string.index("\n")
        for line in string[idx2:].split("\n")[skiplines:]:
            if not line.strip():  # whitespace only
                break
            try:
                units = line[line.index("[")+1:line.index("]")]
            except ValueError:
                raise ValueError("A unit declaration bracketed by [] was"
                                 " not found on the line reading:\n"
                                 "    %s" % line)
            nameval = line[:line.index("[")].split()
            labelstart = line.index("]") + 1
            if labelstart >= len(line):
                label = ""
            else:
                while line[labelstart] == " ":
                    labelstart += 1
                label = line[labelstart:].replace("'", "\\'")
            countstr += variable_declaration(nameval, units, label, line,
                                             errorcatch)
        if declaration_func is None:
            overallstr += countstr
        else:
            overallstr += declaration_func(string, flag, idx2, countstr)
        string = string[idx2+len(flag):]
    return overallstr


PARSETIP = ("Is this line following the format `Name (optional Value) [Units]"
            " (Optional Description)` without any whitespace in the Name or"
            " Value fields?")


def variable_declaration(nameval, units, label, line, errorcatch=True):
    "Turns parsed output into a Variable declaration"
    if len(nameval) > 2:
        raise ValueError("while parsing the line '%s', additional fields"
                         " (separated by whitespace) were found between Value"
                         " '%s' and the Units `%s`. %s"
                         % (line, nameval[1], units, PARSETIP))
    elif len(nameval) == 2:
        out = ("{0} = self.{0} = Variable('{0}', {1}, '{2}', '{3}')")
        out = out.format(nameval[0], nameval[1], units, label)
    elif len(nameval) == 1:
        out = ("{0} = self.{0} = Variable('{0}', '{1}', '{2}')")
        out = out.format(nameval[0], units, label)
    if errorcatch:
        out = """
try:
    {0}
except Exception as e:
    raise ValueError("`"+e.__class__.__name__+": "+str(e)+"` was raised"
                     " while executing the parsed line `{0}`. {1}")
""".format(out, PARSETIP)
    else:
        out = out + "\n"
    return out
