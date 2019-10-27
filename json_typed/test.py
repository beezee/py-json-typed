from adt import Id, Sum2, F1, F2
from dataclasses import dataclass
from json_typed import CustomParseError, ExtendParse
from json_typed import Parse2, Parse3, ListParser, ParseError, parse_str
from json_typed import parse_int, parse_optional, Parser
from typing import List, Tuple

foo = Parser(['foo'], parse_optional(parse_str))
baz = ListParser.FailSlow(['baz'], parse_str)
bar = ListParser.FailFast(['bar'], parse_int)
neg_int = ExtendParse[int, int](parse_int, 
  lambda x: F2(x) if (x < 0) else F1(
    CustomParseError({'constraint': 'negative int', 'value': str(x)})))
quux = Parser(['quux'], neg_int)

@dataclass 
class FooBaz:
  foo: Sum2[None, str]
  bar: List[int]
  baz: List[str]
  baz_err: List[ParseError]
@dataclass
class FBBQ:
  foobar_err: List[ParseError]
  foobar: List[FooBaz]
  quux: int

fbq = Parse2(
  ListParser.FailSlow([], 
    Parse3(foo, bar, baz, lambda x: FooBaz(x[0], x[1], x[2][1], x[2][0])).parseFn())
    .set_path(['foobar']), 
  quux,
  lambda t: FBBQ(t[0][0], t[0][1], t[1]))

if __name__ == '__main__':
  print(fbq.parse('{"quux": null, "foobar": [{"foo": "bar", "bar": [], "baz": ["quux"]}]}'))
  print(fbq.parse('{"quux": -4, "foobar": [1, {"foo": "bar", "bar": [], "baz": [1, "quux"]}]}'))
  print(fbq.parse('{"quux": 33, "foobar": [{"foo": null, "bar": ["a", 2, "3"]}]}'))
  print(fbq.parse('{"quux": null, "foobar": [{"foo": 4, "bar": 3, "baz": [2, "quux", 3]}]}'))
  print(fbq.parse('{"quux": -5, "foobar": [{"bar": [2, 3, 4], "moo": "bar", "baz": []}]}'))

