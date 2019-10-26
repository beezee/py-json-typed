from adt import Id, Sum2
from dataclasses import dataclass
from json_typed import Combine2, parseStr, parseInt, parseOptional, Parser
from typing import Tuple

foo = Parser(['foo'], parseOptional(parseStr()))
bar = Parser(['baz'], parseStr())
quux = Parser(['quux'], parseInt())

@dataclass
class Foo:
  foo: Sum2[None, str]
  bar: str
  quux: int

fbq = Combine2(Combine2(foo, bar, Id[Tuple[Sum2[None, str], str]]())().setPath(['foobar']), quux,
  lambda t: Foo(t[0][0], t[0][1], t[1]))

if __name__ == '__main__':
  print(fbq().parse('{"quux": null, "foobar": {"foo": "bar", "baz": "quux"}}'))
  print(fbq().parse('{"quux": 4, "foobar": {"foo": "bar", "baz": "quux"}}'))
  print(fbq().parse('{"quux": null, "foobar": {"foo": null, "baz": "quux"}}'))
  print(fbq().parse('{"quux": null, "foobar": {"foo": 4, "baz": "quux"}}'))
  print(fbq().parse('{"quux": 5, "foobar": {"moo": "bar", "baz": "quux"}}'))

