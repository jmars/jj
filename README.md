# JJ

WIP Javascript interpeter in restricted python.

A very simple AST interpreter design reusing the esprima parser. I hope to regain the performance loss using the rpython toolchains automatically generated JIT compiler and partial evaluation.

Whether that works is yet to be seen.

If it does, the next step will be to fix enough bugs that it can interpret the esprima parser itself and bootstrap itself. Once that is done I will write a simple ES7 --> RPython translator (without closures) and port the vm to itself, closing the loop.

```
python jj.py
```