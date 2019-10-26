from adt import Id, Sum2
from dataclasses import dataclass
from json_typed import Combine2, ListParser, ParseError, parseStr, parseInt, parseOptional, Parser
from typing import List, Tuple

foo = Parser(['foo'], parseOptional(parseStr()))
baz = ListParser(['baz'], parseStr())
quux = Parser(['quux'], parseInt())

@dataclass
class FBQ:
  foo: Sum2[None, str]
  baz: List[str]
  bazErr: List[ParseError]
  quux: int

fbq = Combine2(Combine2(foo, baz, lambda x: x)().setPath(['foobar']), quux,
  lambda t: FBQ(t[0][0], t[0][1][1], t[0][1][0], t[1]))

if __name__ == '__main__':
  print(fbq().parse('{"quux": null, "foobar": {"foo": "bar", "baz": ["quux"]}}'))
  print(fbq().parse('{"quux": 4, "foobar": {"foo": "bar", "baz": [1, "quux"]}}'))
  print(fbq().parse('{"quux": null, "foobar": {"foo": null, "baz": [2, 3]}}'))
  print(fbq().parse('{"quux": null, "foobar": {"foo": 4, "baz": [2, "quux", 3]}}'))
  print(fbq().parse('{"quux": 5, "foobar": {"moo": "bar", "baz": []}}'))

