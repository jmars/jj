from math import isnan
from json import loads
from subprocess import call
from rpython.rlib.objectmodel import specialize, we_are_translated, always_inline
from rpython.rlib.jit import JitDriver, elidable, unroll_safe, elidable_promote
from rpython.rlib.rweakref import RWeakValueDictionary

# TODO: all types of expressions (--, ++, +, - etc), find with == '

js_jitdriver = JitDriver(greens = ['body'], reds = 'auto')

class Globals(object):
  def __init__(self):
    self.label = ''
    self.exception = u''

@apply
def globals():
  prebuilt = Globals()
  return lambda: prebuilt

def to_primitive(val):
  get_value_of = MemberExpression(False, Constant(val), Constant(W_JSString(u'valueOf')))
  get_to_string = MemberExpression(False, Constant(val), Constant(W_JSString(u'toString')))
  value_of = get_value_of.interpret(None)
  to_string = get_to_string.interpret(None)
  if not isinstance(value_of, W_JSNull):
    return CallExpression(Constant(value_of), []).interpret(None)
  else:
    return CallExpression(Constant(to_string), []).interpret(None)

def to_number(val):
  if isinstance(val, W_JSNumber):
    return val
  elif isinstance(val, W_JSBoolean):
    if val.boolean:
      return W_JSNumber(1.0)
    else:
      return W_JSNumber(0.0)
  elif isinstance(val, W_JSString):
    return W_JSNumber(float(str(val.string)))
  elif isinstance(val, W_JSNull):
    return W_JSNumber(0.0)
  elif isinstance(val, W_JSArray):
    return W_JSNumber(float(len(val.array)))
  else:
    return W_JSNumber(float('NaN'))

def to_string(val):
  if isinstance(val, W_JSString):
    return val
  elif isinstance(val, W_JSNumber):
    return W_JSString(unicode(str(val.number)))
  elif isinstance(val, W_JSBoolean):
    return W_JSString(unicode(str(val.boolean)))
  elif isinstance(val, W_JSNull):
    return W_JSString(u'null')
  elif isinstance(val, W_JSUndefined):
    return W_JSString(u'undefined')
  else:
    return to_string(to_primitive(val))

def same_type(a, b):
  if isinstance(a, W_JSNumber) and isinstance(b, W_JSNumber):
    return True
  elif isinstance(a, W_JSString) and isinstance(b, W_JSString):
    return True
  elif isinstance(a, W_JSNull) and isinstance(b, W_JSNull):
    return True
  elif isinstance(a, W_JSBoolean) and isinstance(b, W_JSBoolean):
    return True
  elif isinstance(a, W_JSFunction) and isinstance(b, W_JSFunction):
    return True
  elif isinstance(a, W_JSArray) and isinstance(b, W_JSArray):
    return True
  elif isinstance(a, W_JSUndefined) and isinstance(b, W_JSUndefined):
    return True
  return False

def double_eq(a, b):
  if a == b: return True
  if same_type(a, b):
    if isinstance(a, W_JSUndefined): return True
    if isinstance(a, W_JSNull): return True
    if isinstance(a, W_JSNumber) and isinstance(b, W_JSNumber):
      if isnan(a.number) or isnan(b.number): return False
      if a.number == b.number: return True
      if a.number == 0.0 and b.number == -0.0: return True
      if a.number == -0.0 and b.number == 0.0: return True
      return False
    elif isinstance(a, W_JSString) and isinstance(b, W_JSString):
      return a.string == b.string
    elif isinstance(a, W_JSBoolean) and isinstance(b, W_JSBoolean):
      return a.boolean == b.boolean
  elif isinstance(a, W_JSNull) and isinstance(b, W_JSUndefined): return True
  elif isinstance(a, W_JSUndefined) and isinstance(b, W_JSNull): return True
  elif isinstance(a, W_JSNumber) and isinstance(b, W_JSString):
    return double_eq(a, to_number(b))
  elif isinstance(a, W_JSString) and isinstance(b, W_JSNumber):
    return double_eq(to_number(a), b)
  elif isinstance(a, W_JSBoolean):
    return double_eq(to_number(a), b)
  elif isinstance(b, W_JSBoolean):
    return double_eq(a, to_number(b))
  elif (isinstance(a, W_JSString) or isinstance(a, W_JSNumber)) and isinstance(b, W_JSObject):
    return double_eq(a, to_primitive(b))
  return False

def looper(node, frame):
  while True:
    js_jitdriver.jit_merge_point(body=node)
    try: node.loop(frame)
    except Break as b:
      string = globals().label
      if string == '': break
      elif node.label != string: raise b
      elif node.label == string:
        if isinstance(node, ForStatement):
          node.update.interpret(frame)
    except Continue: continue

@always_inline
def call_js(function, arguments):
  assert isinstance(function, W_JSFunction)
  scope = Frame(function.frame)
  if not function.this is None:
    scope.this = function.this
  return function.interpret(scope, arguments)

class Return(Exception): pass
class Break(Exception): pass
class Continue(Exception): pass
class IO(Exception): pass
class JSException(Exception): pass

class W_JSIterator(object): pass

class W_JSObjectIterator(W_JSIterator):
  def __init__(self, obj):
    self.keys = [key for key in obj.data]
    self.i = 0

  def next(self):
    if self.i >= len(self.keys): raise StopIteration()
    val = self.keys[self.i]
    self.i = self.i + 1
    return W_JSString(val.string)

class W_JSArrayIterator(W_JSIterator):
  def __init__(self, arr):
    self.arr = arr.array
    self.i = 0

  def next(self):
    if self.i >= len(self.arr): raise StopIteration()
    val = self.arr[self.i]
    self.i = self.i + 1
    return val

class W_Object(object):
  def __init__(self):
    self.n_val = 0
    self.s_val = u''
    self.content = []

  def eq(self, w_other):
    raise NotImplementedError('eq not supported by this class')

  def neq(self, w_other):
    raise NotImplementedError('neq not supported by this class')

  def gt(self, w_other):
    raise NotImplementedError('gt not supported by this class')

  def lt(self, w_other):
    raise NotImplementedError('lt not supported by this class')

  def ge(self, w_other):
    raise NotImplementedError('ge not supported by this class')

  def le(self, w_other):
    raise NotImplementedError('le not supported by this class')

  def is_true(self):
    return True

  def to_str(self):
    raise NotImplementedError('to_str not supported by this class')

  def clone(self):
    raise NotImplementedError('clone not supported by this class')

  def get_val(self, key):
    raise NotImplementedError('get_val not supported by this class')

  def hash(self):
    raise NotImplementedError('hash not supported by this class')

# Objects
class W_JSObject(object):
  def __init__(self, init = None):
    if init is None:
      self.data = {}
    else:
      assert isinstance(init, dict)
      self.data = init

  def get(self, key):
    if key in self.data:
      return self.data[key]
    else:
      return W_JSUndefined()

  def set(self, key, val):
    assert isinstance(val, W_JSObject)
    self.data[key] = val

  def __iter__(self):
    return W_JSObjectIterator(self)

class W_JSExceptionObject(W_JSObject):
  def __init__(self, error):
    self.error = error

  def get(self, key):
    return self

  def set(self, key, value):
    return self

class W_JSNumber(W_JSObject):
  def __init__(self, number):
    self.number = number

  def get(self, key):
    return self

  def set(self, key, value):
    return self

class W_JSString(W_JSObject):
  def __init__(self, string):
    self.string = string

  def get(self, key):
    return self

  def set(self, key, value):
    return self

class W_JSArray(W_JSObject):
  def __init__(self, array):
    self.array = array

  def __iter__(self):
    return W_JSArrayIterator(self)

  def get_i(self, key):
    i = int(to_number(key).number)
    try:
      return self.array[i]
    except IndexError:
      return self.get(s(str(to_string(key).string)))

class W_JSFunction(W_JSObject):
  def __init__(self, params, body, frame):
    self.params = params
    self.body = body
    self.this = None
    self.frame = frame
    self.data = {}

  def interpret(self, frame, args):
    i = 0
    for param in self.params:
      frame.set(param, args[i])
      i = i + 1
    try:
      self.body.interpret(frame)
    except Return:
      return frame.ret

  def bind(self, frame, this, args):
    closure = Frame(self.frame)
    params = this.params[:]
    params.reverse()
    assert len(params) >= len(args)
    for arg in args:
      frame.set(params.pop(), arg)
    params.reverse()
    func = W_JSFunction(params, self.body, closure)
    func.this = this
    return func

class W_JSBoolean(W_JSObject):
  def __init__(self, boolean):
    self.boolean = boolean

  def get(self, key):
    return self

  def set(self, key, value):
    return self

class W_JSNull(W_JSObject):
  def get(self, key):
    return self

  def set(self, key, value):
    return self

class W_JSUndefined(W_JSObject):
  def get(self, key):
    return self

  def set(self, key, value):
    return self

def make_jsfunction(func):
  class W_CustomFunction(W_JSFunction):
    def __init__(self, frame):
      W_JSFunction.__init__(self, [], BlockStatement([EmptyStatement()]), frame)
  W_CustomFunction.interpret = func
  return W_CustomFunction

@apply
def Prim_Proxy():
  def factory(self, frame, args):
    assert len(args) == 2
    target = args[0]
    handler = args[1]
    obj = W_JSObject()
    for key in internal_symbols:
      func = handler.get(internal_symbols[key])
      # TODO
      if not isinstance(func, W_JSFunction): continue
      #
      assert isinstance(func, W_JSFunction)
      f = func.bind(frame, handler, [target])
      obj.set(s(key.string), f)
    return obj

  return make_jsfunction(factory)

@apply
def Prim_Array():
  def factory(self, frame, args):
    return W_JSArray(args)

  return make_jsfunction(factory)

@apply
def Prim_Object():
  def factory(self, frame, args):
    return W_JSObject()

  return make_jsfunction(factory)

@apply
def Prim_Function():
  def factory(self, frame, args):
    return W_JSUndefined()

  return make_jsfunction(factory)

"""
Reflect.apply()
Calls a target function with arguments as specified by the args parameter. See also Function.prototype.apply().
Reflect.construct()
 The new operator as a function. Equivalent to calling new target(...args).
Reflect.defineProperty()
Similar to Object.defineProperty(). Returns a Boolean.
Reflect.deleteProperty()
The delete operator as a function. Equivalent to calling delete target[name].
Reflect.enumerate()
Like the for...in loop. Returns an iterator with the enumerable own and inherited properties of the target object.
Reflect.get()
A function that returns the value of properties.
Reflect.getOwnPropertyDescriptor()
Similar to Object.getOwnPropertyDescriptor(). Returns a property descriptor of the given property if it exists on the object,  undefined otherwise.
Reflect.getPrototypeOf()
Same as Object.getPrototypeOf().
Reflect.has()
The in operator as function. Returns a boolean indicating whether an own or inherited property exists.
Reflect.isExtensible()
Same as Object.isExtensible().
Reflect.ownKeys()
Returns an array of strings with own (not inherited) property keys.
Reflect.preventExtensions()
Similar to Object.preventExtensions(). Returns a Boolean.
Reflect.set()
A function that assigns values to properties. Returns a Boolean that is true if the update was successful.
Reflect.setPrototypeOf()
A function that sets the prototype of an object.
"""

# TODO: use plain dicts and lists for primitive reflection

@apply
def reflection():
  def apply(self, frame, args):
    assert len(args) == 3
    [target, thisArg, argsArray] = args
    assert isinstance(target, W_JSFunction)
    target.this = thisArg
    assert isinstance(argsArray, W_JSArray)
    return call_js(target, argsArray)

  def construct(self, frame, args):
    assert len(args) == 2
    [target, argsArray] = args
    assert isinstance(target, W_JSFunction)
    target.this = W_JSObject()
    assert isinstance(argsArray, W_JSArray)
    call_js(target, argsArray)
    return target.this

  def defineProperty(self, frame, args):
    assert len(args) == 3
    [target, prop, desc] = args
    assert isinstance(target, W_JSObject)
    # TODO: support properties in W_JSObject


  def deleteProperty(self, frame, args): pass
  def enumerate(self, frame, args): pass
  def get(self, frame, args): pass
  def getOwnPropertyDescriptor(self, frame, args): pass
  def getPrototypeOf(self, frame, args): pass
  def has(self, frame, args): pass
  def isExtensible(self, frame, args): pass
  def ownKeys(self, frame, args): pass
  def preventExtensions(self, frame, args): pass
  def set(self, frame, args): pass
  def setPrototypeOf(self, frame, args): pass

  return {
    u'apply' : make_jsfunction(apply),
    u'construct' : make_jsfunction(construct),
    u'defineProperty' : make_jsfunction(defineProperty)
  }

def make_objectspace():
  g = Frame(None)
  g.set(W_JSString(u'null'), W_JSNull())
  g.set(W_JSString(u'undefined'), W_JSUndefined())
  g.set(W_JSString(u'true'), W_JSBoolean(True))
  g.set(W_JSString(u'false'), W_JSBoolean(False))

  g.set(W_JSString(u'Function'), Prim_Function(g))
  g.set(W_JSString(u'Object'), Prim_Object(g))
  g.set(W_JSString(u'Array'), Prim_Array(g))
  g.set(W_JSString(u'Proxy'), Prim_Proxy(g))
  g.set(W_JSString(u'Reflect'), reflection)
  return g

call(['node', 'bootstrap'])
esprima_ast = loads(open('esprima.json', 'r').read())

def JSON_to_JS(tree):
  "NOT RPYTHON"
  if isinstance(tree, dict):
    return W_JSObject(
      init = { k : JSON_to_JS(v) for k, v in tree.iteritems() }
    )
  elif isinstance(tree, list):
    return W_JSArray([JSON_to_JS(x) for x in tree])
  elif isinstance(tree, bool):
    return W_JSBoolean(tree)
  elif isinstance(tree, float):
    return W_JSNumber(tree)
  elif isinstance(tree, int):
    return W_JSNumber(float(tree))
  elif isinstance(tree, long):
    return W_JSNumber(float(tree))
  elif isinstance(tree, str):
    return W_JSString(unicode(tree))
  elif isinstance(tree, unicode):
    return W_JSString(tree)
  elif tree is None:
    return W_JSNull()
  else:
    print tree
    raise Exception("Unknown type " + str(type(tree)))

#-------

class Frame(object):
  def __init__(self, parent):
    self.locals = {}
    self.parent = parent
    self.ret = W_JSNull()
    self.this = None

  @unroll_safe
  @always_inline
  @specialize.call_location()
  def get(self, name):
    assert isinstance(name, W_JSString)
    frame = self
    while not name.string in frame.locals:
      frame = frame.parent
      if frame is None: return W_JSNull()
    return frame.locals[name.string]

  @always_inline
  @unroll_safe
  @specialize.call_location()
  def set(self, name, value):
    assert isinstance(name, W_JSString)
    frame = self
    while True:
      if name.string in frame.locals:
        frame.locals[name.string] = value
        break
      elif not frame.parent is None:
        frame = frame.parent
      else:
        frame.locals[name.string] = value
        break

# PYO -> JSO -> ASTInterp
# JSO -> ASTInterp

class Node(object): pass
class Statement(Node): pass
class Expression(Node): pass

class Program(Statement):
  def __init__(self, body):
    self.body = body

  @unroll_safe
  def interpret(self, frame):
    for stmt in self.body:
      stmt.interpret(frame)
    return frame

class ExpressionStatement(Statement):
  def __init__(self, expression):
    self.expression = expression

  def interpret(self, frame):
    self.expression.interpret(frame)

def rshift(val, n):
  return (val % 0x100000000) >> n

class BinaryExpression(Expression):
  def __init__(self, op, left, right):
    assert isinstance(left, Expression)
    assert isinstance(right, Expression)
    assert isinstance(op, str)
    self.op = op
    self.left = left
    self.right = right

  def interpret(self, frame):
    if self.op == '==':
      l = self.left.interpret(frame)
      r = self.right.interpret(frame)
      return W_JSBoolean(double_eq(l, r))
    elif self.op == '!=':
      l = self.left.interpret(frame)
      r = self.right.interpret(frame)
      return W_JSBoolean(not double_eq(l, r))
    elif self.op == '===':
      l = self.left.interpret(frame)
      r = self.right.interpret(frame)
      if not same_type(l, r):
        return W_JSBoolean(False)
      else:
        return W_JSBoolean(double_eq(l, r))
    elif self.op == '!==':
      l = self.left.interpret(frame)
      r = self.right.interpret(frame)
      if same_type(l, r):
        return W_JSBoolean(False)
      else:
        return W_JSBoolean(not double_eq(l, r))
    elif self.op == '<':
      l = to_number(self.left.interpret(frame))
      r = to_number(self.right.interpret(frame))
      assert isinstance(l, W_JSNumber)
      assert isinstance(r, W_JSNumber)
      return W_JSBoolean(l.number < r.number)
    elif self.op == '<=':
      l = to_number(self.left.interpret(frame))
      r = to_number(self.right.interpret(frame))
      assert isinstance(l, W_JSNumber)
      assert isinstance(r, W_JSNumber)
      return W_JSBoolean(l.number <= r.number)
    elif self.op == '>':
      l = to_number(self.left.interpret(frame))
      r = to_number(self.right.interpret(frame))
      assert isinstance(l, W_JSNumber)
      assert isinstance(r, W_JSNumber)
      return W_JSBoolean(l.number > r.number)
    elif self.op == '>=':
      l = to_number(self.left.interpret(frame))
      r = to_number(self.right.interpret(frame))
      assert isinstance(l, W_JSNumber)
      assert isinstance(r, W_JSNumber)
      return W_JSBoolean(l.number >= r.number)
    elif self.op == '<<':
      l = to_number(self.left.interpret(frame))
      r = to_number(self.right.interpret(frame))
      assert isinstance(l, W_JSNumber)
      assert isinstance(r, W_JSNumber)
      return W_JSNumber(float(int(l.number) << int(r.number)))
    elif self.op == '>>':
      l = to_number(self.left.interpret(frame))
      r = to_number(self.right.interpret(frame))
      assert isinstance(l, W_JSNumber)
      assert isinstance(r, W_JSNumber)
      return W_JSNumber(float(int(l.number) >> int(r.number)))
    elif self.op == '>>>':
      l = to_number(self.left.interpret(frame))
      r = to_number(self.right.interpret(frame))
      assert isinstance(l, W_JSNumber)
      assert isinstance(r, W_JSNumber)
      return W_JSNumber(float(rshift(int(l.number), int(r.number))))
    elif self.op == '+':
      l = to_number(self.left.interpret(frame))
      r = to_number(self.right.interpret(frame))
      assert isinstance(l, W_JSNumber)
      assert isinstance(r, W_JSNumber)
      return W_JSNumber(l.number + r.number)
    elif self.op == '-':
      l = to_number(self.left.interpret(frame))
      r = to_number(self.right.interpret(frame))
      assert isinstance(l, W_JSNumber)
      assert isinstance(r, W_JSNumber)
      return W_JSNumber(l.number - r.number)
    elif self.op == '*':
      l = to_number(self.left.interpret(frame))
      r = to_number(self.right.interpret(frame))
      assert isinstance(l, W_JSNumber)
      assert isinstance(r, W_JSNumber)
      return W_JSNumber(l.number * r.number)
    elif self.op == '/':
      l = to_number(self.left.interpret(frame))
      r = to_number(self.right.interpret(frame))
      assert isinstance(l, W_JSNumber)
      assert isinstance(r, W_JSNumber)
      return W_JSNumber(l.number / r.number)
    elif self.op == '%':
      l = to_number(self.left.interpret(frame))
      r = to_number(self.right.interpret(frame))
      assert isinstance(l, W_JSNumber)
      assert isinstance(r, W_JSNumber)
      return W_JSNumber(l.number % r.number)
    elif self.op == '|':
      l = to_number(self.left.interpret(frame))
      r = to_number(self.right.interpret(frame))
      assert isinstance(l, W_JSNumber)
      assert isinstance(r, W_JSNumber)
      return W_JSNumber(l.number | r.number)
    elif self.op == '^':
      l = to_number(self.left.interpret(frame))
      r = to_number(self.right.interpret(frame))
      assert isinstance(l, W_JSNumber)
      assert isinstance(r, W_JSNumber)
      return W_JSNumber(l.number ^ r.number)
    elif self.op == '&':
      l = to_number(self.left.interpret(frame))
      r = to_number(self.right.interpret(frame))
      assert isinstance(l, W_JSNumber)
      assert isinstance(r, W_JSNumber)
      return W_JSNumber(l.number & r.number)
    else:
      return W_JSNull()

"""
enum BinaryOperator {
    "==" | "!=" | "===" | "!=="
         | "<" | "<=" | ">" | ">="
         | "<<" | ">>" | ">>>"
         | "+" | "-" | "*" | "/" | "%"
         | "|" | "^" | "&" | "in"
         | "instanceof" | ".."
}
"""

class Constant(Expression):
  def __init__(self, value):
    assert isinstance(value, W_JSObject)
    self.value = value

  @elidable_promote()
  def interpret(self, frame):
    return self.value

class Identifier(Expression):
  def __init__(self, name):
    self.name = name

  def interpret(self, frame):
    return frame.get(self.name)

class VariableDeclaration(Statement):
  def __init__(self, kind, declarations):
    self.kind = kind
    self.declarations = declarations

  @unroll_safe
  def interpret(self, frame):
    for decl in self.declarations:
      decl.interpret(frame)

class AssignmentExpression(Expression):
  def __init__(self, typ, left, right):
    self.type = typ
    self.left = left
    self.right = right

  def interpret(self, frame):
    r = self.right.interpret(frame)
    if isinstance(r, W_JSFunction):
      r.this = None
    if isinstance(self.left, Identifier):
      frame.set(self.left.name, r)
    elif isinstance(self.left, MemberExpression):
      self.left.assign(frame, r)
    #if self.type == '=':
    #  frame.set(self.left, r)
    return r

class VariableDeclarator(object):
  def __init__(self, name, init):
    self.name = name
    self.init = init

  def interpret(self, frame):
    if self.init is None:
      frame.set(self.name, W_JSNull())
    else:
      frame.set(self.name, self.init.interpret(frame))

class BlockStatement(Statement):
  def __init__(self, body):
    self.body = body

  @unroll_safe
  def interpret(self, frame):
    for stmt in self.body:
      stmt.interpret(frame)

class FunctionDeclaration(Statement):
  def __init__(self, name, params, body):
    self.name = name
    self.params = params
    self.body = body

  def interpret(self, frame):
    func = W_JSFunction(self.params, self.body, frame)
    frame.set(self.name, func)

class ReturnStatement(Statement):
  def __init__(self, argument):
    self.argument = argument

  def interpret(self, frame):
    v = self.argument.interpret(frame)
    frame.ret = v
    raise Return()

class CallExpression(Expression):
  def __init__(self, target, args):
    self.target = target
    self.args = args

  def interpret(self, frame):
    func = self.target.interpret(frame)
    args = [arg.interpret(frame) for arg in self.args]
    return call_js(func, args)

class ForStatement(Statement):
  def __init__(self, init, test, update, body):
    self.init = init
    self.test = test
    self.update = update
    self.body = body
    self.label = ''

  def interpret(self, frame):
    self.init.interpret(frame)
    looper(self, frame)

  def loop(self, frame):
    test_expr = self.test.interpret(frame)
    assert isinstance(test_expr, W_JSBoolean)
    if not test_expr.boolean:
      globals().label = ''
      raise Break()
    try: self.body.interpret(frame)
    except Continue:
      self.update.interpret(frame)
      globals().label = ''
      raise Continue()
    except Break as b: raise b
    self.update.interpret(frame)

class WhileStatement(Statement):
  def __init__(self, test, body):
    self.test = test
    self.body = body
    self.label = ''

  def interpret(self, frame):
    looper(self, frame)

  def loop(self, frame):
    try:
      test_expr = self.test.interpret(frame)
      assert isinstance(test_expr, W_JSBoolean)
      if not test_expr.boolean:
        globals().label = ''
        raise Break()
      self.body.interpret(frame)
    except Break as b: raise b
    except Continue as c: raise c

class DoWhileStatement(Statement):
  def __init__(self, test, body):
    self.test = test
    self.body = body
    self.label = ''

  def interpret(self, frame):
    self.body.interpret(frame)
    looper(self, frame)

  def loop(self, frame):
    try:
      test_expr = self.test.interpret(frame)
      assert isinstance(test_expr, W_JSBoolean)
      if not test_expr.boolean:
        globals().label = ''
        raise Break()
      self.body.interpret(frame)
    except Break as b: raise b
    except Continue as c: raise c

class ForInStatement(Statement):
  def __init__(self, left, right, body):
    self.left = left
    self.right = right
    self.body = body
    self.label = ''

  # TODO: make this use the looper
  @unroll_safe
  def interpret(self, frame):
    name = self.left.declarations[0].name
    self.left.interpret(frame)
    obj = self.right.interpret(frame)
    assert isinstance(obj, W_JSObject)
    for key in obj:
      frame.set(name, key)
      self.body.interpret(frame)

class UpdateExpression(Expression):
  def __init__(self, operator, argument):
    self.operator = operator
    self.argument = argument

  def interpret(self, frame):
    arg = self.argument.interpret(frame)
    if self.operator == '++':
      assert isinstance(arg, W_JSNumber)
      res = W_JSNumber(arg.number + 1)
      frame.set(self.argument.name, res)
      return res

class BreakStatement(Statement):
  def __init__(self, label):
    self.label = label

  def interpret(self, frame):
    if not self.label is None:
      globals().label = self.label
    else:
      globals().label = ''
    raise Break()

class IfStatement(Statement):
  def __init__(self, test, consequent, alternate):
    self.test = test
    self.consequent = consequent
    self.alternate = alternate

  def interpret(self, frame):
    t = self.test.interpret(frame)
    assert isinstance(t, W_JSBoolean)
    if t.boolean:
      self.consequent.interpret(frame)
    else:
      self.alternate.interpret(frame)

class Property(object):
  def __init__(self, key, value):
    self.key = key
    self.value = value

  def interpret(self, frame):
    if isinstance(self.key, Identifier):
      return (self.key.name, self.value.interpret(frame))
    else:
      return (to_string(self.key.interpret(frame)), self.value.interpret(frame))

class ObjectExpression(Expression):
  def __init__(self, properties):
    self.properties = properties

  def interpret(self, frame):
    constructor = frame.get(W_JSString(u'Object'))
    obj = call_js(constructor, [])
    for prop in self.properties:
      (k, v) = prop.interpret(frame)
      obj.set(s(str(k.string)), v)
    return obj

class FunctionExpression(Expression):
  def __init__(self, params, body):
    self.params = params
    self.body = body

  def interpret(self, frame):
    func = W_JSFunction(self.params, self.body, frame)
    return func

class ArrowFunctionExpression(Expression):
  def __init__(self, params, body):
    self.params = params
    self.body = body

  def interpret(self, frame):
    func = W_JSFunction(self.params, self.body, frame)
    # TODO: Fix lexical this binding
    #func.this = frame.get(W_JSString(u'global'))
    return func

class MemberExpression(Expression):
  def __init__(self, computed, object, property):
    self.computed = computed
    self.object = object
    self.property = property

  def interpret(self, frame):
    o = self.object.interpret(frame)
    k = None
    if self.computed:
      k = self.property.interpret(frame)
    else:
      k = self.property.name
    v = None
    if isinstance(o, W_JSArray):
      v = o.get_i(k)
    else:
      v = o.get(key_store.get(to_string(k).string))
    if isinstance(v, W_JSFunction):
      v.this = o
    return v

  def assign(self, frame, value):
    o = self.object.interpret(frame)
    k = None
    if self.computed:
      k = self.property.interpret(frame)
    else:
      k = self.property.name
    o.set(key_store.get(to_string(k).string), value)
    return value

  def remove(self, frame):
    pass

class ThisExpression(Expression):
  def interpret(self, frame):
    return frame.this

class ContinueStatement(Statement):
  def __init__(self, label):
    self.label = label

  def interpret(self, frame):
    if not self.label is None:
      globals().label = self.label
    raise Continue()

class ArrayExpression(Expression):
  def __init__(self, elements):
    self.elements = elements

  def interpret(self, frame):
    args = [x.interpret(frame) for x in self.elements]
    constructor = frame.get(W_JSString(u'Array'))
    return call_js(constructor, args)

class SequenceExpression(Expression):
  def __init__(self, expressions):
    self.expressions = expressions

  def interpret(self, frame):
    assert len(self.expressions) >= 2
    for expr in self.expressions[:-1]:
      expr.interpret(frame)
    return self.expressions[-1].interpret(frame)

class ConditionalExpression(Expression):
  def __init__(self, test, consequent, alternate):
    self.test = test
    self.consequent = consequent
    self.alternate = alternate

  def interpret(self, frame):
    result = self.test.interpret(frame)
    assert isinstance(result, W_JSBoolean)
    if result.boolean:
      return self.consequent.interpret(frame)
    else:
      return self.alternate.interpret(frame)

class UnaryExpression(Expression):
  def __init__(self, operator, argument):
    self.operator = operator
    self.argument = argument

  # TODO: handle delete etc
  def interpret(self, frame):
    val = self.argument.interpret(frame)
    if self.operator == '-':
      assert isinstance(val, W_JSNumber)
      return W_JSNumber(-(val.number))

@elidable
def noop(): pass

class EmptyStatement(Statement):
  def __init__(self): pass
  def interpret(self, frame): noop()

class LogicalExpression(Expression):
  def __init__(self, operator, left, right):
    self.operator = operator
    self.left = left
    self.right = right

  def interpret(self, frame):
    l = self.left.interpret(frame)
    assert isinstance(l, W_JSBoolean)
    r = self.right.interpret(frame)
    assert isinstance(r, W_JSBoolean)
    if self.operator == '||':
      return W_JSBoolean(l.boolean and r.boolean)

class SwitchCase(object):
  def __init__(self, test, consequent):
    self.test = test
    self.consequent = consequent

  # TODO: generate eq expression
  def check(self, frame, value):
    val = self.test.interpret(frame)
    if double_eq(val, value):
      return True
    else:
      return False

  def interpret(self, frame):
    self.consequent.interpret(frame)

class SwitchStatement(Statement):
  def __init__(self, discriminant, cases):
    self.discriminant = discriminant
    self.cases = cases
    self.label = ''

  @unroll_safe
  def interpret(self, frame):
    matched = False
    value = self.discriminant.interpret(frame)
    for case in self.cases:
      if not matched:
        matched = case.check(frame, value)
      if matched:
        try: case.interpret(frame)
        except Break: break

class WithStatement(Statement):
  def __init__(self, object, body):
    self.object = object
    self.body = body

  def interpret(self, frame):
    scope = Frame(frame)
    obj = self.object.interpret(frame)
    assert isinstance(obj, W_JSObject)
    scope.locals = obj.data
    self.body.interpret(scope)

class NewExpression(Expression):
  def __init__(self, callee, arguments):
    self.callee = callee
    self.arguments = arguments

  def interpret(self, frame):
    c = self.callee.interpret(frame)
    f = c.get(internal_symbols['construct'])
    assert isinstance(f, W_JSFunction)
    args = [x.interpret(frame) for x in self.arguments]
    return 
    """
    p = c.get(s('prototype'))
    o = W_JSObject(p)
    c.this = o
    r = CallExpression(Constant(c), self.arguments).interpret(frame)
    if r is None:
      return o
    else:
      return r
    """

class CatchClause(object):
  def __init__(self, param, body):
    self.param = param
    self.body = body

  def interpret(self, frame, error):
    frame.set(self.param.name, JSExceptionObject(error))
    self.body.interpret(frame)

class TryStatement(Statement):
  def __init__(self, block, handler):
    self.block = block
    self.handler = handler

  def interpret(self, frame):
    try:
      self.block.interpret(frame)
    except JSException:
      e = globals().exception
      self.handler.interpret(frame, e)

class ThrowStatement(Statement):
  def __init__(self, argument):
    self.argument = argument

  def interpret(self, frame):
    e = self.argument.interpret(frame)
    assert isinstance(e, W_JSString)
    globals().exception = e.string
    raise JSException()

#----

visitors = {}

def visit(node):
  v = node.get(u'type')
  assert isinstance(v, W_JSString)
  t = str(v.string)
  m = visitors[t]
  return m(node)

def visit_Program(node):
  v = node.get(u'body')
  assert isinstance(v, W_JSArray)
  val = [visit(x) for x in v]
  return Program(val)

visitors['Program'] = visit_Program

# Statements
def visit_ExpressionStatement(node):
  v = node.get(u'expression')
  return ExpressionStatement(visit(v))

visitors['ExpressionStatement'] = visit_ExpressionStatement

def visit_BlockStatement(node):
  b = node.get(u'body')
  assert isinstance(b, W_JSArray)
  return BlockStatement([visit(x) for x in b])

visitors['BlockStatement'] = visit_BlockStatement

def visit_FunctionDeclaration(node):
  n = node.get(u'id').get(u'name')
  assert isinstance(n, W_JSString)
  p = node.get(u'params')
  assert isinstance(p, W_JSArray)
  params = [x.get(u'name') for x in p]
  b = visit(node.get(u'body'))
  return FunctionDeclaration(n, params, b)

visitors['FunctionDeclaration'] = visit_FunctionDeclaration

def visit_VariableDeclarator(node):
  n = node.get(u'id').get(u'name')
  assert isinstance(n, W_JSString)
  i = node.get(u'init')
  if isinstance(i, W_JSNull):
    return VariableDeclarator(n, None)
  else:
    return VariableDeclarator(n, visit(i))

visitors['VariableDeclarator'] = visit_VariableDeclarator

def visit_VariableDeclaration(node):
  k = node.get(u'kind')
  assert isinstance(k, W_JSString)
  d = node.get(u'declarations')
  assert isinstance(d, W_JSArray)
  ds = [visit(x) for x in d]
  return VariableDeclaration(str(k.string), ds)

visitors['VariableDeclaration'] = visit_VariableDeclaration

def visit_IfStatement(node):
  t = visit(node.get(u'test'))
  c = visit(node.get(u'consequent'))
  a = visit(node.get(u'alternate'))
  return IfStatement(t, c, a)

visitors['IfStatement'] = visit_IfStatement

def visit_ForStatement(node):
  i = visit(node.get(u'init'))
  t = visit(node.get(u'test'))
  u = visit(node.get(u'update'))
  b = visit(node.get(u'body'))
  return ForStatement(i, t, u, b)

visitors['ForStatement'] = visit_ForStatement

def visit_WhileStatement(node):
  t = visit(node.get(u'test'))
  b = visit(node.get(u'body'))
  return WhileStatement(t, b)

visitors['WhileStatement'] = visit_WhileStatement

def visit_DoWhileStatement(node):
  b = visit(node.get(u'body'))
  t = visit(node.get(u'test'))
  return DoWhileStatement(t, b)

visitors['DoWhileStatement'] = visit_DoWhileStatement

def visit_ForInStatement(node):
  l = visit(node.get(u'left'))
  r = visit(node.get(u'right'))
  b = visit(node.get(u'body'))
  return ForInStatement(l, r, b)

visitors['ForInStatement'] = visit_ForInStatement

def visit_ReturnStatement(node):
  a = node.get(u'argument')
  return ReturnStatement(visit(a))

visitors['ReturnStatement'] = visit_ReturnStatement

def visit_BreakStatement(node):
  l = node.get(u'label')
  if isinstance(l, W_JSNull):
    return BreakStatement(None)
  elif isinstance(l, W_JSObject):
    ll = l.get(u'name')
    assert isinstance(ll, W_JSString)
    return BreakStatement(str(ll.string))

visitors['BreakStatement'] = visit_BreakStatement

def visit_CatchClause(node):
  p = visit(node.get(u'param'))
  b = visit(node.get(u'body'))
  return CatchClause(p, b)

visitors['CatchClause'] = visit_CatchClause

def visit_TryStatement(node):
  b = visit(node.get(u'block'))
  h = visit(node.get(u'handler'))
  return TryStatement(b, h)

visitors['TryStatement'] = visit_TryStatement

def visit_ThrowStatement(node):
  a = visit(node.get(u'argument'))
  return ThrowStatement(a)

visitors['ThrowStatement'] = visit_ThrowStatement

def visit_SwitchCase(node):
  t = visit(node.get(u'test'))
  c = BlockStatement([visit(x) for x in node.get(u'consequent')])
  return SwitchCase(t, c)

visitors['SwitchCase'] = visit_SwitchCase

def visit_SwitchStatement(node):
  d = visit(node.get(u'discriminant'))
  c = [visit(x) for x in node.get(u'cases')]
  return SwitchStatement(d, c)

visitors['SwitchStatement'] = visit_SwitchStatement

def visit_ContinueStatement(node):
  l = node.get(u'label')
  if isinstance(l, W_JSNull):
    return ContinueStatement(None)
  elif isinstance(l, W_JSObject):
    ll = l.get(u'name')
    assert isinstance(ll, W_JSString)
    return ContinueStatement(str(ll.string))

visitors['ContinueStatement'] = visit_ContinueStatement

def visit_LabeledStatement(node):
  l = node.get(u'label').get(u'name')
  assert isinstance(l, W_JSString)
  ll = str(l.string)
  b = visit(node.get(u'body'))
  if isinstance(b, ForStatement):
    b.label = ll
    return b
  elif isinstance(b, WhileStatement):
    b.label = ll
    return b
  elif isinstance(b, DoWhileStatement):
    b.label = ll
    return b
  elif isinstance(b, ForInStatement):
    b.label = ll
    return b

visitors['LabeledStatement'] = visit_LabeledStatement

def visit_WithStatement(node):
  o = visit(node.get(u'object'))
  b = visit(node.get(u'body'))
  return WithStatement(o, b)

visitors['WithStatement'] = visit_WithStatement

def visit_EmptyStatement(node):
  return EmptyStatement()

visitors['EmptyStatement'] = visit_EmptyStatement

def visit_DebuggerStatement(node):
  return EmptyStatement()

visitors['DebuggerStatement'] = visit_DebuggerStatement

# Expressions
def visit_AssignmentExpression(node):
  t = node.get(u'operator')
  assert isinstance(t, W_JSString)
  l = visit(node.get(u'left'))
  r = visit(node.get(u'right'))
  return AssignmentExpression(str(t.string), l, r)

visitors['AssignmentExpression'] = visit_AssignmentExpression

def visit_FunctionExpression(node):
  p = node.get(u'params')
  assert isinstance(p, W_JSArray)
  params = [x.get(u'name') for x in p]
  b = visit(node.get(u'body'))
  return FunctionExpression(params, b)

visitors['FunctionExpression'] = visit_FunctionExpression

def visit_ArrowFunctionExpression(node):
  p = node.get(u'params')
  assert isinstance(p, W_JSArray)
  params = [x.get(u'name') for x in p]
  b = visit(node.get(u'body'))
  return ArrowFunctionExpression(params, b)

visitors['ArrowFunctionExpression'] = visit_ArrowFunctionExpression

def visit_Identifier(node):
  n = node.get(u'name')
  assert isinstance(n, W_JSString)
  return Identifier(n)

visitors['Identifier'] = visit_Identifier

def visit_Literal(node):
  v = node.get(u'value')
  return Constant(v)

visitors['Literal'] = visit_Literal

def visit_UnaryExpression(node):
  o = str(node.get(u'operator').string)
  a = visit(node.get(u'argument'))
  return UnaryExpression(o, a)

visitors['UnaryExpression'] = visit_UnaryExpression

def visit_BinaryExpression(node):
  o = node.get(u'operator')
  assert isinstance(o, W_JSString)
  l = node.get(u'left')
  r = node.get(u'right')
  op = str(o.string)
  return BinaryExpression(op, visit(l), visit(r))

visitors['BinaryExpression'] = visit_BinaryExpression

def visit_LogicalExpression(node):
  o = node.get(u'operator')
  assert isinstance(o, W_JSString)
  l = visit(node.get(u'left'))
  r = visit(node.get(u'right'))
  return LogicalExpression(str(o.string), l, r)

visitors['LogicalExpression'] = visit_LogicalExpression

def visit_MemberExpression(node):
  o = visit(node.get(u'object'))
  p = visit(node.get(u'property'))
  c = node.get(u'computed')
  assert isinstance(c, W_JSBoolean)
  return MemberExpression(c.boolean, o, p)

visitors['MemberExpression'] = visit_MemberExpression

def visit_CallExpression(node):
  ce = visit(node.get(u'callee'))
  a = node.get(u'arguments')
  assert isinstance(a, W_JSArray)
  args = [visit(x) for x in a]
  return CallExpression(ce, args)

visitors['CallExpression'] = visit_CallExpression

def visit_NewExpression(node):
  c = visit(node.get(u'callee'))
  a = node.get(u'arguments')
  assert isinstance(a, W_JSArray)
  args = [visit(x) for x in a]
  return NewExpression(c, args)

visitors['NewExpression'] = visit_NewExpression

def visit_ThisExpression(node):
  return ThisExpression() 

visitors['ThisExpression'] = visit_ThisExpression

def visit_Property(node):
  k = visit(node.get(u'key'))
  v = visit(node.get(u'value'))
  return Property(k, v)

visitors['Property'] = visit_Property

def visit_ObjectExpression(node):
  p = node.get(u'properties')
  assert isinstance(p, W_JSArray)
  ps = [visit(x) for x in p]
  return ObjectExpression(ps)

visitors['ObjectExpression'] = visit_ObjectExpression

def visit_UpdateExpression(node):
  o = node.get(u'operator')
  assert isinstance(o, W_JSString)
  a = visit(node.get(u'argument'))
  return UpdateExpression(str(o.string), a)

visitors['UpdateExpression'] = visit_UpdateExpression

def visit_ArrayExpression(node):
  e = node.get(u'elements')
  assert isinstance(e, W_JSArray)
  el = [visit(x) for x in e]
  return ArrayExpression(el)

visitors['ArrayExpression'] = visit_ArrayExpression

def visit_ConditionalExpression(node):
  t = visit(node.get(u'test'))
  c = visit(node.get(u'consequent'))
  a = visit(node.get(u'alternate'))
  return ConditionalExpression(t, c, a)

visitors['ConditionalExpression'] = visit_ConditionalExpression

def visit_SequenceExpression(node):
  e = node.get(u'expressions')
  assert isinstance(e, W_JSArray)
  ex = [visit(x) for x in e]
  return SequenceExpression(ex)

visitors['SequenceExpression'] = visit_SequenceExpression

test = JSON_to_JS(esprima_ast)
ast = visit(test)

def entry_point(argv):
  global_f = make_objectspace()
  frame = ast.interpret(global_f)
  b = frame.get(W_JSString(u'b'))
  assert isinstance(b, W_JSNumber)
  print b.number
  return 0

def target(*args):
  return entry_point, None

if __name__ == '__main__':
  import sys
  entry_point(sys.argv)
